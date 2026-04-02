import glob
import json
import PIL.Image as Image
from PIL import ImageFilter
import os
import math

import torch
from torch.utils.data import Dataset
from torchvision import transforms
from pytorch3d.io import load_objs_as_meshes
from pytorch3d.renderer import (
    look_at_view_transform,
    FoVPerspectiveCameras,
    MeshRasterizer,
    RasterizationSettings
)


class SimulationPairedImageDataset(Dataset):
    def __init__(self, adv_dir, orig_dir, parms_dir, obj_dir, feed_size, img_size=(1024, 320), device='cuda'):
        self.device = torch.device(device)
        self.my_dir = adv_dir
        self.adv_paths = sorted(glob.glob(f"{adv_dir}/*.png"))
        self.orig_paths = sorted(glob.glob(f"{orig_dir}/*.png"))
        assert len(self.adv_paths) == len(self.orig_paths), "Number of adversarial and original images must match"

        feed_height = feed_size[0]
        feed_width = feed_size[1]

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((feed_height, feed_width)),
             transforms.Lambda(lambda x: x.to(self.device))])

        with open(os.path.join(parms_dir), "r", encoding='utf-8') as f:
            self.params = json.load(f)

        self.mesh = load_objs_as_meshes([obj_dir], device=self.device)

        # 加载相机参数
        self.image_size_x = img_size[0]  # CARLA 中的图像宽度
        self.image_size_y = img_size[1]  # 图像高度
        self.fov_horizontal_deg = 120.0  # CARLA 中给出的水平视场角
        self.fov_vertical_rad = 2.0 * math.atan(
            math.tan(math.radians(self.fov_horizontal_deg) / 2.0) * (self.image_size_y / self.image_size_x))
        self.fov_vertical_deg = math.degrees(self.fov_vertical_rad)
        self.cameras = FoVPerspectiveCameras(fov=self.fov_vertical_deg, zfar=100, device=self.device)

        self.raster_settings = RasterizationSettings(
            image_size=(self.image_size_y, self.image_size_x),
            max_faces_per_bin=5000,
            bin_size=0
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings,
        )

    def __len__(self):
        return len(self.adv_paths)

    def __getitem__(self, idx):
        # 从深度图生成掩膜 (非背景区域为1)
        position_name = os.path.splitext(os.path.basename(self.adv_paths[idx]))[0]
        position_params = self.params[position_name]
        R, T = look_at_view_transform(dist=position_params['dist'], elev=position_params['elev'], azim=position_params['azim'],
                                      degrees=False,
                                      device=self.device)
        self.cameras.R = R
        self.cameras.T = T
        fragments = self.rasterizer(self.mesh)
        vehicle_mask = (fragments.pix_to_face[0, ..., 0] > -1).float().unsqueeze(0)

        adv_img = Image.open(self.adv_paths[idx]).convert('RGB')
        orig_img = Image.open(self.orig_paths[idx]).convert('RGB')
        adv_tensor = self.transform(adv_img)
        orig_tensor = self.transform(orig_img)
        return position_name, position_params, adv_tensor, orig_tensor, vehicle_mask

class DetectionImageDataset(Dataset):
    def __init__(self, my_dir, parms_dir, obj_dir, img_size=(1024, 320), device='cuda'):
        self.device = torch.device(device)
        self.my_dir = my_dir
        self.my_paths = sorted(glob.glob(f"{my_dir}/*.png"))
        with open(os.path.join(parms_dir), "r", encoding='utf-8') as f:
            self.params = json.load(f)

        self.mesh = load_objs_as_meshes([obj_dir], device=self.device)

        # 加载相机参数
        self.image_size_x = img_size[0]  # CARLA 中的图像宽度
        self.image_size_y = img_size[1]  # 图像高度
        self.fov_horizontal_deg = 120.0  # CARLA 中给出的水平视场角
        self.fov_vertical_rad = 2.0 * math.atan(
            math.tan(math.radians(self.fov_horizontal_deg) / 2.0) * (self.image_size_y / self.image_size_x))
        self.fov_vertical_deg = math.degrees(self.fov_vertical_rad)
        self.cameras = FoVPerspectiveCameras(fov=self.fov_vertical_deg, zfar=100, device=self.device)

        self.raster_settings = RasterizationSettings(
            image_size=(self.image_size_y, self.image_size_x),
            max_faces_per_bin=5000,
            bin_size=0
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings,
        )

    @staticmethod
    def mask_to_bbox(mask: torch.Tensor):
        if mask.dim() == 3:
            mask = mask.squeeze(0)

        ys, xs = torch.nonzero(mask, as_tuple=True)
        x1, y1 = xs.min().item(), ys.min().item()
        x2, y2 = xs.max().item(), ys.max().item()

        return torch.tensor([[x1, y1, x2, y2]])

    def __len__(self):
        return len(self.my_paths)

    def __getitem__(self, idx):
        position_name = os.path.splitext(os.path.basename(self.my_paths[idx]))[0]
        position_path = self.my_paths[idx]
        position_params = self.params[position_name]
        R, T = look_at_view_transform(dist=position_params['dist'], elev=position_params['elev'], azim=position_params['azim'],
                                      degrees=False,
                                      device=self.device)
        self.cameras.R = R
        self.cameras.T = T
        fragments = self.rasterizer(self.mesh)
        vehicle_mask = (fragments.pix_to_face[0, ..., 0] > -1).float().unsqueeze(0)
        bbox = self.mask_to_bbox(vehicle_mask)
        return position_path, position_name, bbox


class SegmentationImageDataset(Dataset):
    def __init__(self, my_dir, parms_dir, obj_dir, img_size=(1024, 320), device='cuda'):
        self.device = torch.device(device)
        self.my_dir = my_dir
        self.my_paths = sorted(glob.glob(f"{my_dir}/*.png"))
        with open(os.path.join(parms_dir), "r", encoding='utf-8') as f:
            self.params = json.load(f)

        self.mesh = load_objs_as_meshes([obj_dir], device=self.device)

        # 加载相机参数
        self.image_size_x = img_size[0]  # CARLA 中的图像宽度
        self.image_size_y = img_size[1]  # 图像高度
        self.fov_horizontal_deg = 120.0  # CARLA 中给出的水平视场角
        self.fov_vertical_rad = 2.0 * math.atan(
            math.tan(math.radians(self.fov_horizontal_deg) / 2.0) * (self.image_size_y / self.image_size_x))
        self.fov_vertical_deg = math.degrees(self.fov_vertical_rad)
        self.cameras = FoVPerspectiveCameras(fov=self.fov_vertical_deg, zfar=100, device=self.device)

        self.raster_settings = RasterizationSettings(
            image_size=(self.image_size_y, self.image_size_x),
            max_faces_per_bin=5000,
            bin_size=0
        )
        self.rasterizer = MeshRasterizer(
            cameras=self.cameras,
            raster_settings=self.raster_settings,
        )

    def __len__(self):
        return len(self.my_paths)

    def __getitem__(self, idx):
        position_name = os.path.splitext(os.path.basename(self.my_paths[idx]))[0]
        position_path = self.my_paths[idx]
        position_params = self.params[position_name]
        R, T = look_at_view_transform(dist=position_params['dist'], elev=position_params['elev'], azim=position_params['azim'],
                                      degrees=False,
                                      device=self.device)
        self.cameras.R = R
        self.cameras.T = T
        fragments = self.rasterizer(self.mesh)
        vehicle_mask = (fragments.pix_to_face[0, ..., 0] > -1).float().unsqueeze(0)

        return position_path, position_name, vehicle_mask


class PhysicalPairedImageDataset(Dataset):
    def __init__(self, adv_dir, adv_mask_dir, orig_dir, orig_mask_dir, feed_size, blur_sigma=0, image_size=(1920, 1080), device='cuda'):
        self.device = torch.device(device)
        self.adv_paths = sorted(glob.glob(f"{adv_dir}/*.png"))
        self.adv_mask_paths = sorted(glob.glob(f"{adv_mask_dir}/*.png"))
        self.orig_paths = sorted(glob.glob(f"{orig_dir}/*.png"))
        self.orig_mask_paths = sorted(glob.glob(f"{orig_mask_dir}/*.png"))

        self.image_size_y = image_size[1]
        self.image_size_x = image_size[0]

        self.blur_sigma = blur_sigma

        feed_height = feed_size[0]
        feed_width = feed_size[1]

        assert len(self.adv_paths) == len(self.orig_paths), "Number of adversarial and original images must match"

        self.transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Resize((feed_height, feed_width)),
             transforms.Lambda(lambda x: x.to(self.device))])

        self.mask_transform = transforms.Compose(
            [transforms.ToTensor(),
             transforms.Lambda(lambda x: x.to(self.device))])

    @staticmethod
    def gaussian_blur(img : Image.Image, sigma):
        return img.filter(ImageFilter.GaussianBlur(radius=sigma))

    def __len__(self):
        return len(self.adv_paths)

    def __getitem__(self, idx):
        # 从深度图生成掩膜 (非背景区域为1)
        name = os.path.splitext(os.path.basename(self.adv_paths[idx]))[0]

        adv_img = Image.open(self.adv_paths[idx]).convert('RGB')
        adv_mask = Image.open(self.adv_mask_paths[idx]).convert('1')
        orig_img = Image.open(self.orig_paths[idx]).convert('RGB')
        orig_mask = Image.open(self.orig_mask_paths[idx]).convert('1')

        adv_img = self.gaussian_blur(adv_img, self.blur_sigma)
        orig_img = self.gaussian_blur(orig_img, self.blur_sigma)

        adv_tensor = self.transform(adv_img)
        adv_mask_tensor = self.mask_transform(adv_mask)
        orig_tensor = self.transform(orig_img)
        orig_mask_tensor = self.mask_transform(orig_mask)


        return name, adv_tensor, adv_mask_tensor, orig_tensor, orig_mask_tensor
