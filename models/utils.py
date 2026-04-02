import torch
import os
import numpy as np
import matplotlib as mpl
import matplotlib.cm as cm
from PIL import Image

class DepthModelWrapper(torch.nn.Module):
    def __init__(self, encoder=None, decoder=None, model=None) -> None:
        super(DepthModelWrapper, self).__init__()
        self.encoder = encoder
        self.decoder = decoder
        self.model = model

    def forward(self, input_image):
        if self.model is not None:
            outputs = self.model(input_image)
        else:
            features = self.encoder(input_image)
            outputs = self.decoder(features)

        return outputs[("disp", 0)]

def disp_to_depth(disp, min_depth, max_depth):
    """Convert network's sigmoid output into depth prediction
    The formula for this conversion is given in the 'additional considerations'
    section of the paper.
    """
    min_disp = 1 / max_depth
    max_disp = 1 / min_depth
    scaled_disp = min_disp + (max_disp - min_disp) * disp
    depth = 1 / scaled_disp
    return scaled_disp, depth

def disp_to_image(orig_tensor, adv_tensor, orig_disp, adv_disp, resize_func, path, name):
    # make path
    os.makedirs(path, exist_ok=True)

    # convert tensor to array
    orig_tensor_np = resize_func(orig_tensor * 255).byte().cpu().numpy().transpose(1, 2, 0)
    adv_tensor_np = resize_func(adv_tensor * 255).byte().cpu().numpy().transpose(1, 2, 0)
    orig_disp_np = resize_func(orig_disp).squeeze().cpu().numpy()
    adv_disp_np = resize_func(adv_disp).squeeze().cpu().numpy()

    # mapping
    vmax = np.percentile(orig_disp_np, 95)
    normalizer = mpl.colors.Normalize(vmin=orig_disp_np.min(), vmax=vmax)
    mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')

    orig_disp_colored = (mapper.to_rgba(orig_disp_np)[:, :, :3] * 255).astype(np.uint8)
    adv_disp_colored = (mapper.to_rgba(adv_disp_np)[:, :, :3] * 255).astype(np.uint8)

    # Create an image grid (2x2)
    top = np.hstack((orig_tensor_np, adv_tensor_np))
    bottom = np.hstack((orig_disp_colored, adv_disp_colored))
    final_image = np.vstack((top, bottom))

    # Save the final image
    Image.fromarray(final_image).save(os.path.join(path, f'{name}.png'))

def get_depth(disp, model_name, resize_func):
    if model_name == 'monodepth2' or model_name == 'depthhints':
        STEREO_SCALE_FACTOR = 5.4
        disp_resized = resize_func(disp)
        metric_depth = disp_to_depth(disp_resized, 0.1, 100)[1] * STEREO_SCALE_FACTOR
        return torch.clamp(metric_depth, 0, 80)

    elif model_name == 'monovit' or model_name == 'robustdepth':
        STEREO_SCALE_FACTOR = 5.4
        disp_resized = resize_func(disp)
        metric_depth = disp_to_depth(disp_resized, 0.1, 80)[1] * STEREO_SCALE_FACTOR
        return torch.clamp(metric_depth, 0, 80)

    else:
        raise ValueError(f"Unsupported MDE model: {model_name}")

def get_mean_depth_diff(adv_disp1, ben_disp2, scene_car_mask, model_name, resize_func, reduce=True):
    dep1_adv = get_depth(adv_disp1, model_name, resize_func)
    dep2_ben = get_depth(ben_disp2, model_name, resize_func)
    diff = (dep1_adv - dep2_ben) * scene_car_mask
    if reduce:
        mean_diff = diff.sum() / (scene_car_mask.sum() + 1e-8)
        return mean_diff
    else:
        return diff.sum(dim=[1, 2, 3])


def get_affected_ratio(disp1, disp2, scene_car_mask, model_name, resize_func, threshold=10, reduce=True):
    dep1 = get_depth(disp1, model_name, resize_func)
    dep2 = get_depth(disp2, model_name, resize_func)
    affected = scene_car_mask * ((dep1 - dep2) > threshold).float()
    if reduce:
        affected_ratio = affected.sum() / (scene_car_mask.sum() + 1e-8)
        return affected_ratio
    else:
        return affected.sum(dim=[1, 2, 3])
