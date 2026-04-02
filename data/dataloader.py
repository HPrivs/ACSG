import math
import numpy as np
import os.path
import json
from PIL import Image
from typing import Optional

import torch
from torch import Tensor
from torch.utils.data import Dataset
from torch.nn.functional import grid_sample, softmax, interpolate
from torchvision.transforms.functional import gaussian_blur, resized_crop, InterpolationMode
from torchvision.transforms.transforms import RandomResizedCrop
import torchvision.io as io

from pytorch3d.io import load_objs_as_meshes, load_obj
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    DirectionalLights,
    Materials,
    RasterizationSettings,
    MeshRenderer,
    MeshRasterizer,
    TexturesUV,
)

from pytorch3d.renderer.mesh.shader import ShaderBase, phong_shading, softmax_rgb_blend, BlendParams
from pytorch3d.renderer.mesh.rasterizer import Fragments
from pytorch3d.renderer.utils import TensorProperties
from pytorch3d.structures.meshes import Meshes
from pytorch3d.ops import interpolate_face_attributes
from pytorch3d.common.datatypes import Device


class HyBridSoftPhongShader(ShaderBase):
    def __init__(
            self,
            device: Device = "cuda",
            cameras: Optional[TensorProperties] = None,
            lights: Optional[TensorProperties] = None,
            materials: Optional[Materials] = None,
            blend_params: Optional[BlendParams] = None,
            use_img_aug: bool = True,
    ):
        super().__init__(device=device, cameras=cameras, lights=lights, materials=materials, blend_params=blend_params)
        self.use_img_aug = use_img_aug

    def forward(self, fragments: Fragments, meshes: Meshes, **kwargs) -> tuple[Tensor, Tensor, Tensor]:
        cameras = super()._get_cameras(**kwargs)

        texture = kwargs.get("adv_texture")
        mask_sampler = kwargs.get("mask_sampler")
        detail_sampler = kwargs.get("detail_sampler")
        lights = kwargs.get("lights", self.lights)
        materials = kwargs.get("materials", self.materials)
        blend_params = kwargs.get("blend_params", self.blend_params)
        names = kwargs.get("names")

        znear = kwargs.get("znear", getattr(cameras, "znear", 1.0))
        zfar = kwargs.get("zfar", getattr(cameras, "zfar", 50.0))

        # 获取车辆掩膜
        vehicle_masks = (fragments.pix_to_face[..., 0] >= 0).float().unsqueeze(1)

        texels_detail = detail_sampler.sample_textures(fragments)
        texels_mask = mask_sampler.sample_textures(fragments)

        texels_triplanar = self._get_triplanar_texels(fragments, meshes, texture)
        texels = torch.lerp(input=texels_detail, end=texels_triplanar, weight=texels_mask)

        colors = phong_shading(
            meshes=meshes,
            fragments=fragments,
            texels=texels,
            lights=lights,
            cameras=cameras,
            materials=materials,
        )

        img = softmax_rgb_blend(
            colors, fragments, blend_params, znear, zfar
        )

        return img, fragments.zbuf, vehicle_masks

    def _get_triplanar_texels(self, fragments: Fragments, meshes: Meshes, texture: torch.Tensor) -> torch.Tensor:
        N, H_out, W_out, K = fragments.pix_to_face.shape
        _N, C, H_in, W_in = texture.shape

        contrast = 5.0

        verts = meshes.verts_packed()
        faces = meshes.faces_packed()
        verts_normals = meshes.verts_normals_packed()
        verts_attrs = torch.cat([verts, verts_normals], dim=1)
        faces_attrs = verts_attrs[faces]
        interp_attrs = interpolate_face_attributes(
            fragments.pix_to_face,
            fragments.bary_coords,
            faces_attrs,
        )

        world_pos = interp_attrs[..., :3]  # (N, H, W, K, 3)
        world_normals = interp_attrs[..., 3:]  # (N, H, W, K, 3)

        verts = meshes.verts_packed()
        verts_center = verts.mean(dim=0)
        center = verts_center
        centered_pos = world_pos - center

        if self.use_img_aug:
            max_offset = 0.5
            # scale_factor = 1.28
            scale_factor = 1.28 + (torch.rand(1, device=centered_pos.device) * 2.0 - 1.0) * 0.25
            random_offset = (torch.rand((N, 1, 1, 1, 3), device=world_pos.device) * 2.0 - 1.0) * max_offset

            scaled_pos = (centered_pos / scale_factor) + random_offset
        else:
            scaled_pos = centered_pos

        scaled_pos = scaled_pos.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 3)
        world_normals = world_normals.permute(0, 3, 1, 2, 4).reshape(N * K, H_out, W_out, 3)

        wrapped_coords = torch.remainder(scaled_pos, 1.0)

        norm_coords = wrapped_coords * 2.0 - 1.0
        coords_xy = norm_coords[..., [0, 1]]
        coords_yz = norm_coords[..., [1, 2]]
        coords_xz = norm_coords[..., [0, 2]]

        coords_xy = torch.nan_to_num(coords_xy, nan=0.0, posinf=0.0, neginf=0.0)
        coords_yz = torch.nan_to_num(coords_yz, nan=0.0, posinf=0.0, neginf=0.0)
        coords_xz = torch.nan_to_num(coords_xz, nan=0.0, posinf=0.0, neginf=0.0)

        coords_xy.mul_(torch.tensor([1, -1], device=coords_xy.device))
        coords_yz.mul_(torch.tensor([1, -1], device=coords_yz.device))
        coords_xz.mul_(torch.tensor([1, -1], device=coords_xz.device))

        texture_maps = (
            texture[None, ...]
            .expand(K, -1, -1, -1, -1)
            .transpose(0, 1)
            .reshape(N * K, C, H_in, W_in)
        )

        tex_xy = grid_sample(texture_maps, coords_xy, mode='bilinear', padding_mode='border', align_corners=False)
        tex_yz = grid_sample(texture_maps, coords_yz, mode='bilinear', padding_mode='border', align_corners=False)
        tex_xz = grid_sample(texture_maps, coords_xz, mode='bilinear', padding_mode='border', align_corners=False)

        weights = softmax(torch.abs(world_normals) * contrast, dim=-1)
        weights = weights.permute(0, 3, 1, 2)

        texels = (
                weights[:, 0:1] * tex_yz +
                weights[:, 1:2] * tex_xz +
                weights[:, 2:3] * tex_xy
        )

        texels = texels.reshape(N, K, C, H_out, W_out).permute(0, 3, 4, 1, 2)

        return texels


def apply_sunny_bloom_like(
        rendered_objects: torch.Tensor,
        names: list,
        vehicle_masks: torch.Tensor,
        device: torch.device,
) -> torch.Tensor:
    threshold = 0.6
    bloom_strength = 0.3
    blur_kernel_size = 5
    blur_sigma = 1.0

    original_size = (rendered_objects.shape[2], rendered_objects.shape[3])
    downscale_factor = 8
    bloom_size = (original_size[0] // downscale_factor, original_size[1] // downscale_factor)

    N, C, H, W = rendered_objects.shape
    bloom_mask = torch.tensor([1.0 if 'sunny' in n else 0.0 for n in names], device=device).view(N, 1, 1, 1)

    brightness = torch.logsumexp(rendered_objects * 10, dim=1, keepdim=True) / 10
    bright_areas = torch.clamp(brightness - threshold, min=0.0)
    bright_areas = bright_areas * bloom_mask * vehicle_masks

    bright_areas_small = interpolate(bright_areas, size=bloom_size, mode='bilinear', align_corners=False)
    bright_blur_small = gaussian_blur(bright_areas_small, kernel_size=[blur_kernel_size, blur_kernel_size], sigma=[blur_sigma, blur_sigma])
    bright_blur = interpolate(bright_blur_small, size=original_size, mode='bilinear', align_corners=False)

    output = rendered_objects + bright_blur * bloom_strength
    output = torch.clamp_(output, 0.0, 1.0)

    return output


def apply_fog_effect(
        rendered_objects: torch.Tensor,
        zbuf: torch.Tensor,
        vehicle_masks: torch.Tensor,
        names: list,
        device: torch.device
) -> torch.Tensor:
    N = rendered_objects.shape[0]

    fog_mask = torch.tensor([1.0 if 'foggy' in n else 0.0 for n in names], device=device).view(N, 1, 1, 1)
    fog_color = torch.tensor([1.0, 1.0, 1.0], device=device).view(1, 3, 1, 1).expand(N, -1, -1, -1)
    fog_density = torch.rand(N, device=device) * 0.1 + 0.05
    fog_density = fog_density.view(N, 1, 1, 1) * fog_mask
    distance = zbuf[..., 0].unsqueeze(1)

    max_fog_opacity = 0.5
    fog_factor = 1.0 - torch.exp(-distance * fog_density)
    fog_factor = torch.clamp_(fog_factor, 0.0, max_fog_opacity) * vehicle_masks

    foggy_objects = rendered_objects * (1.0 - fog_factor) + fog_color * fog_factor

    return foggy_objects


class CarlaDataset(Dataset):
    """
    position_name: 相机位姿参数文件
    img_size: 图像宽高
    obj_name: 车辆网格体文件
    mask_name: 伪装掩膜文件
    """

    def __init__(self, data_dir='./dataset/rgb', position_dir='./dataset/', img_size=(1280, 960), obj_name='./Asset/lexus_hs.obj',
                 uv_mask_name='./Asset/lexus_mask.jpg', apply_phys_aug=True, apply_FBSG=True, apply_trip_aug=True, device=torch.device("cuda:0")):
        self.device = device
        self.data_dir = data_dir
        self.position_dir = position_dir
        self.apply_phys_aug = apply_phys_aug
        self.apply_img_aug = apply_trip_aug
        self.apply_FBSG = apply_FBSG

        # 加载obj
        self.mesh = load_objs_as_meshes([obj_name], device=self.device)

        self.verts, self.faces, self.aux = load_obj(obj_name, load_textures=True, texture_wrap='repeat', device=self.device)

        self.verts_uvs = self.aux.verts_uvs
        self.faces_uvs = self.faces.textures_idx

        # 读取纹理
        self.orig_texture = list(self.aux.texture_images.values())[0][None].to(self.device)  # [1, 1024, 1024, 3]
        self.adv_texture = torch.sigmoid(torch.randn(1, 3, 512, 512).to(self.device))

        # 读取纹理mask
        self.uv_mask = np.array(Image.open(uv_mask_name).convert('L')).astype(np.float32) / 255.0
        self.binary_uv_mask = torch.from_numpy(self.uv_mask).unsqueeze(0).unsqueeze(-1).to(self.device)  # H W C
        self.uv_size_x = self.binary_uv_mask.shape[1]
        self.uv_size_y = self.binary_uv_mask.shape[2]

        # 纹理采样器
        self.detail_sampler = TexturesUV(
            maps=self.orig_texture,
            faces_uvs=[self.faces_uvs],
            verts_uvs=[self.verts_uvs]
        ).to(device)

        self.mask_sampler = TexturesUV(
            maps=self.binary_uv_mask,
            faces_uvs=[self.faces_uvs],
            verts_uvs=[self.verts_uvs]
        ).to(device)

        # 加载相机参数
        self.image_size_x = img_size[0]  # CARLA 中的图像宽度
        self.image_size_y = img_size[1]  # 图像高度
        self.fov_horizontal_deg = 120.0  # CARLA 中给出的水平视场角
        self.fov_vertical_rad = 2.0 * math.atan(
            math.tan(math.radians(self.fov_horizontal_deg) / 2.0) * (self.image_size_y / self.image_size_x))
        self.fov_vertical_deg = math.degrees(self.fov_vertical_rad)

        # 加载位姿
        position_name = os.path.join(self.position_dir, 'positions.json')
        with open(position_name, 'r', encoding='utf-8') as f:
            self.loaded_positions = json.load(f)

        # 光照参数
        self.light_color_bases = {
            'sunny': {
                'ambient': torch.tensor([1.0, 0.95, 0.9], device=device),
                'diffuse': torch.tensor([1.0, 0.95, 0.85], device=device),
                'specular': torch.tensor([1.0, 0.98, 0.95], device=device)
            },
            'cloudy': {
                'ambient': torch.tensor([0.95, 0.92, 0.9], device=device),
                'diffuse': torch.tensor([0.9, 0.88, 0.85], device=device),
                'specular': torch.tensor([1.0, 1.0, 1.0], device=device)
            },
            'foggy': {
                'ambient': torch.tensor([0.8, 0.8, 0.9], device=device),
                'diffuse': torch.tensor([0.9, 0.9, 1.0], device=device),
                'specular': torch.tensor([1.0, 1.0, 1.0], device=device)
            },
            'rainy': {
                'ambient': torch.tensor([0.8, 0.78, 0.75], device=device),
                'diffuse': torch.tensor([0.8, 0.78, 0.75], device=device),
                'specular': torch.tensor([1.0, 1.0, 1.0], device=device)
            },
            'night': {
                'ambient': torch.tensor([0.89, 0.91, 1.0], device=device),
                'diffuse': torch.tensor([1.0, 0.85, 0.7], device=device),
                'specular': torch.tensor([1.0, 1.0, 1.0], device=device)
            },
            'default': {
                'ambient': torch.tensor([1.0, 1.0, 1.0], device=device),
                'diffuse': torch.tensor([1.0, 1.0, 1.0], device=device),
                'specular': torch.tensor([1.0, 1.0, 1.0], device=device)
            }
        }

        self.files = sorted(os.listdir(os.path.join(data_dir)))

        self.cameras_dict = {}
        self.valid_indices = []
        for file in self.files:
            data_name = os.path.splitext(file)[0]

            if data_name not in self.loaded_positions:
                print(f"Warning: {data_name} missing, skipping...")
                continue

            rgb_path = self.get_rgb_path(os.path.join(self.data_dir), data_name)

            try:
                Image.open(rgb_path).verify()
            except Exception as e:
                print(f"Warning: {rgb_path} is corrupted, skipping. Error: {e}")
                continue

            R, T = look_at_view_transform(
                dist=self.loaded_positions[data_name]['dist'],
                elev=self.loaded_positions[data_name]['elev'],
                azim=self.loaded_positions[data_name]['azim'],
                degrees=False
            )
            R = R.to(device)
            T = T.to(device)
            light = np.array(self.loaded_positions[data_name]['light'])

            idx = len(self.valid_indices)  # 连续索引
            self.cameras_dict[idx] = {'name': data_name, 'R': R, 'T': T, 'light': light}
            self.valid_indices.append(data_name)

        self.cameras = FoVPerspectiveCameras(fov=self.fov_vertical_deg, zfar=50, device=self.device)
        self.lights = DirectionalLights(device=self.device)

        # create materials for rendering
        self.materials = Materials(
            device=device,
            ambient_color=[[1.0, 1.0, 1.0]],
            diffuse_color=[[1.0, 1.0, 1.0]],
            specular_color=[[1.0, 1.0, 1.0]],
            shininess=500.0
        )

        self.raster_settings = RasterizationSettings(
            image_size=(self.image_size_y, self.image_size_x),
            faces_per_pixel=1,
            max_faces_per_bin=2000,
        )

        self.renderer = MeshRenderer(
            rasterizer=MeshRasterizer(
                cameras=self.cameras,
                raster_settings=self.raster_settings,
            ),
            shader=HyBridSoftPhongShader(
                cameras=self.cameras,
                lights=self.lights,
                device=self.device,
                use_img_aug=self.apply_img_aug
            )
        )

    def __getitem__(self, index):
        cam_params = self.cameras_dict[index]
        names = cam_params['name']
        self.cameras.R = cam_params['R']
        self.cameras.T = cam_params['T']

        if self.apply_phys_aug:
            self.get_lights_params(cam_params['name'], cam_params['light'].astype(np.float32))

        camouflage_objects, zbuf, vehicle_masks = self.renderer(self.mesh,
                                                                materials=self.materials,
                                                                adv_texture=self.adv_texture,
                                                                detail_sampler=self.detail_sampler,
                                                                mask_sampler=self.mask_sampler,
                                                                names=names)  # B H W C

        camouflage_objects = camouflage_objects[..., :3].permute(0, 3, 1, 2)  # B C H W

        if self.apply_phys_aug:
            camouflage_objects = apply_fog_effect(
                rendered_objects=camouflage_objects,
                zbuf=zbuf,
                vehicle_masks=vehicle_masks,
                names=[names],
                device=self.device
            )

            camouflage_objects = apply_sunny_bloom_like(
                rendered_objects=camouflage_objects,
                vehicle_masks=vehicle_masks,
                names=[names],
                device=self.device
            )

        # 生成对抗场景
        img_path = self.get_rgb_path(self.data_dir, names)
        original_scenes = io.read_image(img_path, mode=io.ImageReadMode.RGB).float().div(255).to(self.device).unsqueeze(0)
        original_objects = (1 - vehicle_masks) + vehicle_masks * original_scenes  # B C H W torch.float32
        camouflage_scenes = (1 - vehicle_masks) * original_scenes + vehicle_masks * camouflage_objects

        original_scenes, camouflage_scenes, vehicle_masks = self.random_resized_crop(original_scenes, camouflage_scenes, vehicle_masks)

        return names, original_objects.squeeze(0), camouflage_objects.squeeze(0), original_scenes, camouflage_scenes, vehicle_masks

    def __len__(self):
        return len(self.valid_indices)

    def set_textures(self, adv_texture):
        self.adv_texture = adv_texture

    def get_lights_params(self, name, direction):
        if 'sunny' in name:
            base = self.light_color_bases.get(name, self.light_color_bases['sunny'])
            ambient_intensity = torch.empty(1, device=self.device).uniform_(0.4, 0.7)
            diffuse_intensity = torch.empty(1, device=self.device).uniform_(0.85, 1.0)
            specular_intensity = torch.empty(1, device=self.device).uniform_(0.8, 1.0)
        elif 'cloudy' in name:
            base = self.light_color_bases.get(name, self.light_color_bases['cloudy'])
            ambient_intensity = torch.empty(1, device=self.device).uniform_(0.6, 0.7)
            diffuse_intensity = torch.empty(1, device=self.device).uniform_(0.5, 0.7)
            specular_intensity = torch.empty(1, device=self.device).uniform_(0.1, 0.2)
        elif 'foggy' in name:
            base = self.light_color_bases.get(name, self.light_color_bases['foggy'])
            ambient_intensity = torch.empty(1, device=self.device).uniform_(0.6, 0.8)
            diffuse_intensity = torch.empty(1, device=self.device).uniform_(0.5, 0.7)
            specular_intensity = torch.empty(1, device=self.device).uniform_(0.2, 0.4)
        elif 'rainy' in name:
            base = self.light_color_bases.get(name, self.light_color_bases['rainy'])
            ambient_intensity = torch.empty(1, device=self.device).uniform_(0.6, 0.7)
            diffuse_intensity = torch.empty(1, device=self.device).uniform_(0.3, 0.4)
            specular_intensity = torch.empty(1, device=self.device).uniform_(0.1, 0.2)
        elif 'night' in name:
            base = self.light_color_bases.get(name, self.light_color_bases['night'])
            ambient_intensity = torch.empty(1, device=self.device).uniform_(0.35, 0.55)
            diffuse_intensity = torch.empty(1, device=self.device).uniform_(0.05, 0.1)
            specular_intensity = torch.empty(1, device=self.device).uniform_(0.0, 0.05)
        else:
            base = self.light_color_bases.get(name, self.light_color_bases['default'])
            ambient_intensity = torch.empty(1, device=self.device).uniform_(0.3, 0.5)
            diffuse_intensity = torch.empty(1, device=self.device).uniform_(0.7, 1.0)
            specular_intensity = torch.empty(1, device=self.device).uniform_(0.8, 1.2)

        ambient_color = base['ambient'] * ambient_intensity
        diffuse_color = base['diffuse'] * diffuse_intensity
        specular_color = base['specular'] * specular_intensity

        self.renderer.shader.lights.direction = torch.from_numpy(direction).unsqueeze(0)
        self.renderer.shader.lights.ambient_color = ambient_color.unsqueeze(0)
        self.renderer.shader.lights.diffuse_color = diffuse_color.unsqueeze(0)
        self.renderer.shader.lights.specular_color = specular_color.unsqueeze(0)


    @staticmethod
    def get_rgb_path(data_dir, data_name):
        for ext in ['jpg', 'png']:
            path = os.path.join(data_dir, f'{data_name}.{ext}')
            if os.path.exists(path):
                return path
        raise FileNotFoundError(f"No file found for {data_name} in jpg/png format.")

    @staticmethod
    def random_resized_crop(img1, img2, mask, output_size=(320, 1024), scale=(0.50, 1.00), ratio=(1.92, 4.48)):
        if img1.dim() == 4: img1 = img1.squeeze(0)
        if img2.dim() == 4: img2 = img2.squeeze(0)
        if mask.dim() == 4: mask = mask.squeeze(0)

        top, left, crop_height, crop_width = RandomResizedCrop.get_params(img1, scale=scale, ratio=ratio)

        img1_cropped = resized_crop(img1, top, left, crop_height, crop_width, output_size, interpolation=InterpolationMode.BILINEAR)
        img2_cropped = resized_crop(img2, top, left, crop_height, crop_width, output_size, interpolation=InterpolationMode.BILINEAR)
        mask_cropped = resized_crop(mask, top, left, crop_height, crop_width, output_size, interpolation=InterpolationMode.NEAREST)

        return img1_cropped, img2_cropped, mask_cropped

