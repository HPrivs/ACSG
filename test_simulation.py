import os
import tqdm
import math

import torch
from torch.utils.data import DataLoader
from torchvision import transforms

from models.utils import disp_to_image, get_mean_depth_diff, get_affected_ratio
from utils.load_model import load_mde_model
from data.dataset import SimulationPairedImageDataset

import matplotlib.pyplot as plt
import numpy as np

def test(loader):
    depth_model.eval()
    WEATHER_LIST = ['sunny', 'cloudy', 'rainy', 'foggy', 'night']
    # 分天气累加字典，每个天气存 Ra_sum, Ed_sum, mask_sum
    weather_stats = {w: {'Ra_sum': 0.0, 'Ed_sum': 0.0, 'mask_sum': 0.0} for w in WEATHER_LIST}
    # 不同距离累加字典，距离范围3-15米，每2米一个区间
    distance_ranges = [(i, i + 2) for i in range(3, 15, 2)]
    distance_stats = {dist: {'Ed_sum': 0.0, 'mask_sum': 0.0} for dist in distance_ranges}
    # 不同角度累加字典，角度范围0-360度
    degree_ranges = [(i, i + 30) for i in range(0, 360, 30)]
    degree_stats = {dist: {'Ed_sum': 0.0, 'mask_sum': 0.0} for dist in degree_ranges}

    # 全局累加
    total_Ra_sum = 0.0
    total_Ed_sum = 0.0
    total_mask_sum = 0.0
    for position_names, position_params, adv_tensor, orig_tensor, vehicle_mask in loader:
        with torch.no_grad():
            original_disp = depth_model(orig_tensor)
            camouflage_disp = depth_model(adv_tensor)

        Ra = get_affected_ratio(camouflage_disp, original_disp, vehicle_mask, model_name=model, resize_func=input_resize_image, reduce=False)
        Ed = get_mean_depth_diff(camouflage_disp, original_disp, vehicle_mask, model_name=model, resize_func=input_resize_image, reduce=False)

        vehicle_mask = vehicle_mask.sum(dim=[1, 2, 3])

        # 保存结果
        for b in range(orig_tensor.size(0)):
            orig_tensor_b = orig_tensor[b]
            adv_tensor_b = adv_tensor[b]
            original_disp_b = original_disp[b]
            camouflage_disp_b = camouflage_disp[b]
            name = position_names[b]

            disp_to_image(orig_tensor_b, adv_tensor_b, original_disp_b, camouflage_disp_b,
                          input_resize_image, f'./simulation/mde/{model}/{exp_name}', name)

        # 全局累加
        total_Ed_sum += Ed.sum().item()
        total_Ra_sum += Ra.sum().item()
        total_mask_sum += vehicle_mask.sum().item()
        for i, name in enumerate(position_names):
            # 统计天气
            for w in WEATHER_LIST:
                if w in name:
                    weather_stats[w]['Ed_sum'] += Ed[i].item()
                    weather_stats[w]['Ra_sum'] += Ra[i].item()
                    weather_stats[w]['mask_sum'] += vehicle_mask[i].item()
                    break

            # 统计距离
            dist = int(math.floor(position_params['dist'][i].item()))
            for start, end in distance_ranges:
                if (start, end) == (13, 15):
                    if start <= dist <= end:
                        distance_stats[(start, end)]['Ed_sum'] += Ed[i].item()
                        distance_stats[(start, end)]['mask_sum'] += vehicle_mask[i].item()
                        break
                else:
                    if start <= dist < end:
                        distance_stats[(start, end)]['Ed_sum'] += Ed[i].item()
                        distance_stats[(start, end)]['mask_sum'] += vehicle_mask[i].item()
                        break

            # 统计角度
            degree = int(name.split('_')[-1])
            for start, end in degree_ranges:
                if start <= degree < end:
                    degree_stats[(start, end)]['Ed_sum'] += Ed[i].item()
                    degree_stats[(start, end)]['mask_sum'] += vehicle_mask[i].item()
                    break

    # 输出统计
    for w in WEATHER_LIST:
        if weather_stats[w]['mask_sum'] > 0:
            avg_Ed = (weather_stats[w]['Ed_sum'] / weather_stats[w]['mask_sum'])
            avg_Ra = (weather_stats[w]['Ra_sum'] / weather_stats[w]['mask_sum'])
            print(f"{w}: Avg Ed = {avg_Ed:.4f}, Avg Ra = {avg_Ra:.4f}")
        else:
            print(f"{w}: No samples")

    # 输出距离统计
    for range_key in distance_stats:
        if distance_stats[range_key]['mask_sum'] > 0:
            avg_Ed = distance_stats[range_key]['Ed_sum'] / distance_stats[range_key]['mask_sum']
            print(f"Distance {range_key}: Avg Ed = {avg_Ed:.4f}")
        else:
            print(f"Distance {range_key}: No samples")

    # 输出角度统计
    for range_key in degree_stats:
        if degree_stats[range_key]['mask_sum'] > 0:
            avg_Ed = degree_stats[range_key]['Ed_sum'] / degree_stats[range_key]['mask_sum']
            print(f"Degree {range_key}: Avg Ed = {avg_Ed:.4f}")
        else:
            print(f"Degree {range_key}: No samples")

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
    model = 'monodepth2'

    depth_model, feed_height, feed_width = load_mde_model(model, backbone='resnet', resolution='HR')


    paired_dataset = SimulationPairedImageDataset(adv_dir=r'./simulation/patch/camou_citroen', orig_dir='./simulation/original/citroen',
                                                  parms_dir='./simulation/original/position_params.json', obj_dir='./Asset/citroen.obj',
                                                  feed_size=(feed_height, feed_width), img_size=(1024, 320))

    exp_name = os.path.basename(os.path.dirname(paired_dataset.my_dir)) + "-" + os.path.basename(paired_dataset.my_dir)

    input_resize_image = transforms.Resize([paired_dataset.image_size_y, paired_dataset.image_size_x])

    dataloader = DataLoader(paired_dataset, batch_size=4, shuffle=False)
    tqdm_test_loader = tqdm.tqdm(dataloader, desc=f"Simulation Test")

    test(tqdm_test_loader)