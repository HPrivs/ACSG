import glob
import sys
import os
import PIL.Image as Image
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
import torch
from torchvision import transforms

sys.path.append(os.path.abspath(''))
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'mde', 'monodepth2')))

from models.utils import DepthModelWrapper, disp_to_depth

def load_mde_model(model_name, resolution='HR', device='cuda'):
    if model_name == 'monodepth2':
        import mde.monodepth2.networks as net_m2
        if resolution == 'HR':
            model_path = os.path.join("mde", "monodepth2", "models", "mono+stereo_1024x320")
        elif resolution == 'MR':
            model_path = os.path.join("mde", "monodepth2", "models", "mono+stereo_640x192")
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        depth_encoder = net_m2.ResnetEncoder(18, False)
        depth_decoder = net_m2.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))

        # extract the height and width of image that this model was trained with
        loaded_dict_enc = torch.load(encoder_path, map_location=device)
        feed_height = loaded_dict_enc['height']
        feed_width = loaded_dict_enc['width']

        filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in depth_encoder.state_dict()}
        depth_encoder.load_state_dict(filtered_dict_enc)
        loaded_dict = torch.load(depth_decoder_path, map_location=device)
        depth_decoder.load_state_dict(loaded_dict)
        depth_model = DepthModelWrapper(depth_encoder, depth_decoder).to(device)

        for param in depth_model.parameters():
            param.requires_grad = False

        return depth_model, feed_height, feed_width


def disp_to_image(disps, name, savetype):
    # 1CHW
    os.makedirs(f'./simulation/test1/colormapped/', exist_ok=True)
    disp_resized_np = input_resize_image(disps[0]).squeeze().cpu().numpy()  # H W
    vmax = np.percentile(disp_resized_np, 95)
    normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
    colormapped_array = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)  # np H W C
    Image.fromarray(colormapped_array).save(f'./simulation/test1/colormapped/{name}_{savetype}.png')


def get_depth(disp, model_name):
    if model_name == 'monodepth2' or model_name == 'depthhints' or model_name == 'manydepth':
        STEREO_SCALE_FACTOR = 5.4
        disp_resized = input_resize_image(disp)
        metric_depth = disp_to_depth(disp_resized, 0.1, 100)[1] * STEREO_SCALE_FACTOR
        return torch.clamp(metric_depth, 0, 80)

    elif model_name == 'monovit' or model_name == 'robustdepth':
        STEREO_SCALE_FACTOR = 5.4
        disp_resized = input_resize_image(disp)
        metric_depth = disp_to_depth(disp_resized, 0.1, 80)[1] * STEREO_SCALE_FACTOR
        return torch.clamp(metric_depth, 0, 80)

    else:
        raise ValueError(f"Unsupported MDE model: {model_name}")


def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask, model_name, reduce=True):
    dep1_adv = get_depth(adv_disp1, model_name)
    dep2_ben = get_depth(ben_disp2, model_name)

    diff = (dep1_adv - dep2_ben) * scene_car_mask
    if reduce:
        mean_diff = diff.sum() / (scene_car_mask.sum() + 1e-8)
        return mean_diff
    else:
        return diff.sum(dim=[1, 2, 3])


def get_affected_ratio(disp1, disp2, scene_car_mask, model_name, threshold=10, reduce=True):
    dep1 = get_depth(disp1, model_name)
    dep2 = get_depth(disp2, model_name)

    affected = scene_car_mask * ((dep1 - dep2) > threshold).float()
    if reduce:
        affected_ratio = affected.sum() / (scene_car_mask.sum() + 1e-8)
        return affected_ratio
    else:
        return affected.sum(dim=[1, 2, 3])


def test(image_path):
    depth_model.eval()

    if os.path.isfile(image_path):
        # Only testing on a single image
        paths = [image_path]
        output_directory = os.path.dirname(image_path)
    elif os.path.isdir(image_path):
        # Searching folder for images
        paths = glob.glob(os.path.join(image_path, '*.jpg'))
        output_directory = os.path.join(image_path, 'predict')
        os.mkdir(output_directory)
    else:
        raise Exception("Can not find image_path: {}".format(image_path))

    print("-> Predicting on {:d} test images".format(len(paths)))

    # PREDICTING ON EACH IMAGE IN TURN
    with torch.no_grad():
        for idx, image_path in enumerate(paths):

            if image_path.endswith("_disp.jpg"):
                # don't try to predict disparity for a disparity image!
                continue

            # Load image and preprocess
            input_image = Image.open(image_path).convert('RGB')
            original_width, original_height = input_image.size
            input_image = input_image.resize((feed_width, feed_height), Image.Resampling.LANCZOS)
            input_image = transforms.ToTensor()(input_image).unsqueeze(0)

            # PREDICTION
            input_image = input_image.to(device)
            disp = depth_model(input_image)

            disp_resized = torch.nn.functional.interpolate(
                disp, (original_height, original_width), mode="bilinear", align_corners=False)

            # Saving numpy file
            output_name = os.path.splitext(os.path.basename(image_path))[0]
            scaled_disp, depth = disp_to_depth(disp, 0.1, 100)
            metric_depth = 5.4 * depth.cpu().numpy()

            # Saving colormapped depth image
            disp_resized_np = disp_resized.squeeze().cpu().numpy()
            vmax = np.percentile(disp_resized_np, 95)
            normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
            mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
            colormapped_im = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
            im = Image.fromarray(colormapped_im)

            name_dest_im = os.path.join(output_directory, "{}_disp.png".format(output_name))
            im.save(name_dest_im)

            print("   Processed {:d} of {:d} images - saved predictions to:".format(
                idx + 1, len(paths)))
            print("   - {}".format(name_dest_im))

    print('-> Done!')



if __name__ == '__main__':
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # 加载MDE模型
    model = 'monodepth2'
    depth_model, feed_height, feed_width = load_mde_model(model, resolution='HR')

    test(r'C:\Users\HPrivs\Desktop\ori.jpg')
