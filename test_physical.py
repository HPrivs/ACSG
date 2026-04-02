import os
import tqdm
import torch

from torch.utils.data import DataLoader
from torchvision import transforms

from models.utils import disp_to_image, get_depth
from utils.load_model import load_mde_model
from data.dataset import PhysicalPairedImageDataset


def get_mean_depth_diff(adv_disp, orig_disp, adv_mask, orig_mask, model_name):
    dep1_adv = get_depth(adv_disp, model_name, input_resize_image)
    dep2_ben = get_depth(orig_disp, model_name, input_resize_image)

    adv = (dep1_adv * adv_mask).sum()
    orig = (dep2_ben * orig_mask).sum()
    diff = adv - orig
    return diff


def get_affected_ratio(disp1, disp2, adv_mask, model_name, threshold=10):
    dep1 = get_depth(disp1, model_name, input_resize_image)
    dep2 = get_depth(disp2, model_name, input_resize_image)

    affected = adv_mask * ((dep1 - dep2) > threshold).float()
    affected = affected.sum()

    return affected


def test(loader):
    depth_model.eval()

    # 全局累加
    total_Ra_sum = 0.0
    total_Ed_sum = 0.0
    total_mask_sum = 0.0

    for names, adv_tensor, adv_mask_tensor, orig_tensor, orig_mask_tensor in loader:
        with torch.no_grad():
            original_disp = depth_model(orig_tensor)
            camouflage_disp = depth_model(adv_tensor)

        Ra = get_affected_ratio(camouflage_disp, original_disp, adv_mask_tensor, model_name=model)
        Ed = get_mean_depth_diff(camouflage_disp, original_disp, adv_mask_tensor, orig_mask_tensor, model_name=model)

        vehicle_mask = adv_mask_tensor.sum()

        # 全局累加
        total_Ed_sum += Ed.item()
        total_Ra_sum += Ra.item()
        total_mask_sum += vehicle_mask.item()

        for b in range(orig_tensor.size(0)):
            orig_tensor_b = orig_tensor[b]
            adv_tensor_b = adv_tensor[b]
            original_disp_b = original_disp[b]
            camouflage_disp_b = camouflage_disp[b]
            name = names[b]

            disp_to_image(orig_tensor_b, adv_tensor_b, original_disp_b, camouflage_disp_b, half_res_resize,
                          os.path.join(frames_path, 'colormapped'), name)

    # 输出整体指标
    if total_mask_sum > 0:
        overall_Ed = total_Ed_sum / total_mask_sum
        overall_Ra = total_Ra_sum / total_mask_sum
        print(f"Overall Avg Ed = {overall_Ed:.4f}, Overall Avg Ra = {overall_Ra:.4f}")
    else:
        print("No samples evaluated.")


if __name__ == '__main__':
    # device
    if torch.cuda.is_available():
        device = torch.device("cuda:0")
    else:
        device = torch.device("cpu")
        print("WARNING: CPU only, this will be slow!")

    # 加载MDE模型
    model = 'monovit'
    depth_model, feed_height, feed_width = load_mde_model(model, resolution='HR')

    frames_path = r'D:\PythonProject\Framework\simulation\素材\物理世界\indoor'
    adv_dir = os.path.join(frames_path, 'frames_adv', 'merged_frames')
    adv_mask_dir = os.path.join(frames_path, 'frames_adv', 'merged_masks')
    orig_dir = os.path.join(frames_path, 'frames', 'merged_frames')
    orig_mask_dir = os.path.join(frames_path, 'frames', 'merged_masks')

    paired_dataset = PhysicalPairedImageDataset(adv_dir=adv_dir, adv_mask_dir=adv_mask_dir, orig_dir=orig_dir, orig_mask_dir=orig_mask_dir,
                                                blur_sigma=0,
                                                feed_size=(feed_height, feed_width))
    input_resize_image = transforms.Resize([paired_dataset.image_size_y, paired_dataset.image_size_x])
    half_res_resize = transforms.Resize([paired_dataset.image_size_y // 2, paired_dataset.image_size_x // 2])
    dataloader = DataLoader(paired_dataset, batch_size=1, shuffle=False)
    tqdm_test_loader = tqdm.tqdm(dataloader, desc=f"Simulation Test")

    test(tqdm_test_loader)
