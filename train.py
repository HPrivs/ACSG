import os
import random
from datetime import datetime
from copy import deepcopy

import PIL.Image as Image
import matplotlib as mpl
import matplotlib.cm as cm
import numpy as np
import torch
import torch.optim as optim
import tqdm
from torch.utils.data import random_split
from torch.utils.data.dataloader import DataLoader
from torch.utils.tensorboard import SummaryWriter
from torchvision import transforms
from torchvision.utils import make_grid

from data.dataloader import CarlaDataset
from models.texture_generator import AdversarialTextureGenerator
from models.utils import get_depth, get_mean_depth_diff, get_affected_ratio
from utils.load_model import load_mde_model
import torch.nn.functional as F

def update_new_config(config: dict, updates: dict) -> dict:
    new_config = deepcopy(config)

    def recursive_update(d, key, value):
        if key in d:
            d[key] = value
            return True
        for subval in d.values():
            if isinstance(subval, dict):
                if recursive_update(subval, key, value):
                    return True
        return False

    for key, value in updates.items():
        if not recursive_update(new_config, key, value):
            raise KeyError(f"Key '{key}' not found in config structure.")

    return new_config


def set_seed(seed=42):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def start_tensorboard(log_dir="runs"):
    import subprocess
    process = subprocess.Popen([
        'tensorboard',
        "--logdir", log_dir,
    ])
    return process


def disp_to_image(disps):
    to_tensor = transforms.ToTensor()
    batch = []

    for i in range(disps.shape[0]):
        disp_resized_np = input_resize_image(disps[i]).squeeze().cpu().numpy()
        vmax = np.percentile(disp_resized_np, 95)
        normalizer = mpl.colors.Normalize(vmin=disp_resized_np.min(), vmax=vmax)
        mapper = cm.ScalarMappable(norm=normalizer, cmap='magma')
        colormapped_array = (mapper.to_rgba(disp_resized_np)[:, :, :3] * 255).astype(np.uint8)
        batch.append(colormapped_array)

    batch = torch.stack([to_tensor(img) for img in batch], dim=0)
    grid = Image.fromarray((make_grid(batch, nrow=4, padding=0).permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))
    return grid


def save_texture(path, name, save_tensor=False):
    os.makedirs(path, exist_ok=True)
    adv_tex = dataset.adv_texture.squeeze(0).permute(1, 2, 0).detach().cpu()
    adv_image = Image.fromarray((adv_tex.numpy() * 255).astype(np.uint8))
    adv_image.save(os.path.join(path, f'{name}_adv_texture.png'))


def save_checkpoint(model, optimizer, path, epoch):
    checkpoint_path = os.path.join(path, 'checkpoint')
    os.makedirs(checkpoint_path, exist_ok=True)
    torch.save({
        'model_state_dict': model.state_dict(),
        'optimizer_state_dict': optimizer.state_dict(),
    }, os.path.join(checkpoint_path, f'ckpt_{epoch}.pt'))


def loss_adv(camouflage_depth, original_depth, vehicle_mask, gamma_min=0.3, gamma_max=2.0, eps=1e-6):
    B, C, H, W = camouflage_depth.shape

    # 计算像素级的相对差异
    rel = torch.log(camouflage_depth + eps) - torch.log(original_depth + eps)
    pixel_weights = torch.zeros_like(rel)

    areas = vehicle_mask.view(B, -1).sum(dim=1).float()
    max_area = areas.max().clamp(min=eps)

    for b in range(B):
        rel_b = rel[b]
        mask_b = vehicle_mask[b] == 1
        rel_masked_b = rel_b[mask_b]
        rel_norm_b = torch.sigmoid(rel_masked_b)

        area_norm = areas[b] / max_area
        gamma_b = gamma_min + (1 - area_norm) * (gamma_max - gamma_min)

        rel_norm_b = (1 - rel_norm_b) ** gamma_b
        pixel_weights[b][mask_b] = rel_norm_b

    weights = pixel_weights

    adv_loss = - torch.sum(rel * vehicle_mask * weights) / (torch.sum(vehicle_mask * weights) + eps)

    return adv_loss


def loss_smooth(img):
    dx = img[:, :, 1:, :] - img[:, :, :-1, :]
    dy = img[:, :, :, 1:] - img[:, :, :, :-1]
    return dx.pow(2).mean() + dy.pow(2).mean()

def validation(loader, models, writer, epoch, val_size, tv_loss_weight, device):
    model.eval()
    total_adv_loss_sum = 0.0
    total_tv_loss_sum = 0.0
    total_loss_sum = 0.0
    total_Ra_sum = 0.0
    total_Ed_sum = 0.0
    total_mask_sum = 0.0
    WEATHER_LIST = ['sunny', 'cloudy', 'foggy', 'rainy', 'night']
    weather_stats = {w: {'Ra_sum': 0.0, 'Ed_sum': 0.0, 'mask_sum': 0.0} for w in WEATHER_LIST}

    for names, original_object, camouflage_object, original_scene, camouflage_scene, vehicle_mask in loader:
        depth_model = random.choice(models)

        with torch.no_grad():
            vehicle_mask = input_resize_mde(vehicle_mask.to(device))

            original_disp = depth_model(input_resize_mde(original_scene.to(device)))
            camouflage_disp = depth_model(input_resize_mde(camouflage_scene.to(device)))

            original_depth = get_depth(original_disp, model_name, input_resize_image)
            camouflage_depth = get_depth(camouflage_disp, model_name, input_resize_image)

            adv_loss = loss_adv(camouflage_depth, original_depth, vehicle_mask)
            tv_loss = (loss_smooth(dataset.adv_texture) * tv_loss_weight)
            total_loss = adv_loss + tv_loss

            Ed = get_mean_depth_diff(camouflage_disp, original_disp, vehicle_mask, model_name=model_name, resize_func=input_resize_mde, reduce=False)
            Ra = get_affected_ratio(camouflage_disp, original_disp, vehicle_mask, model_name=model_name, resize_func=input_resize_mde, reduce=False)
            vehicle_mask = vehicle_mask.sum(dim=[1, 2, 3])
            batch_size = original_scene.size(0)

            total_adv_loss_sum += adv_loss.item() * batch_size
            total_tv_loss_sum += tv_loss.item() * batch_size
            total_loss_sum = total_loss.item() * batch_size
            total_Ra_sum += Ra.sum().item()
            total_Ed_sum += Ed.sum().item()
            total_mask_sum += vehicle_mask.sum().item()

            for b_idx, name in enumerate(names):
                for w in WEATHER_LIST:
                    if w in name:
                        weather_stats[w]['Ed_sum'] += Ed[b_idx].item()
                        weather_stats[w]['Ra_sum'] += Ra[b_idx].item()
                        weather_stats[w]['mask_sum'] += vehicle_mask[b_idx].item()
                        break

        adv_texture = model()
        dataset.set_textures(adv_texture)
        loader.set_postfix_str(f"adv_loss: {adv_loss.item():.6f} tv_loss: {tv_loss.item():.6f}")

    avg_adv_loss = total_adv_loss_sum / val_size
    avg_tv_loss = total_tv_loss_sum / val_size
    avg_total_loss = total_loss_sum / val_size
    avg_Ra = total_Ra_sum / total_mask_sum
    avg_Ed = total_Ed_sum / total_mask_sum
    ra_dict, ed_dict = {}, {}

    for w in WEATHER_LIST:
        if weather_stats[w]['mask_sum'] > 0:
            ed_dict[w] = (weather_stats[w]['Ed_sum'] / weather_stats[w]['mask_sum'])
            ra_dict[w] = (weather_stats[w]['Ra_sum'] / weather_stats[w]['mask_sum'])

    writer.add_scalar('Val/Loss/adv', avg_adv_loss, epoch)
    writer.add_scalar('Val/Loss/tv', avg_tv_loss, epoch)
    writer.add_scalar('Val/Loss/total', avg_total_loss, epoch)
    writer.add_scalar('Val/Metric/Ra', avg_Ra, epoch)
    writer.add_scalar('Val/Metric/Ed', avg_Ed, epoch)
    writer.add_scalars('Weather/Ra', ra_dict, epoch)
    writer.add_scalars('Weather/Ed', ed_dict, epoch)


def train(loader, models, writer, epoch, global_step, exp_directory, tv_loss_weight, device):
    model.train()

    for i, (names, original_object, camouflage_object, original_scene, camouflage_scene, vehicle_mask) in enumerate(loader):
        depth_model = random.choice(models)
        vehicle_mask = input_resize_mde(vehicle_mask.to(device))

        original_disp = depth_model(input_resize_mde(original_scene))
        camouflage_disp = depth_model(input_resize_mde(camouflage_scene))

        original_depth = get_depth(original_disp, model_name, input_resize_image)
        camouflage_depth = get_depth(camouflage_disp, model_name, input_resize_image)

        with torch.no_grad():
            Ed = get_mean_depth_diff(camouflage_disp, original_disp, vehicle_mask, model_name=model_name, resize_func=input_resize_mde, reduce=True)
            Ra = get_affected_ratio(camouflage_disp, original_disp, vehicle_mask, model_name=model_name, resize_func=input_resize_mde, reduce=True)

        # 计算损失函数
        adv_loss = loss_adv(camouflage_depth, original_depth, vehicle_mask)
        tv_loss = (loss_smooth(dataset.adv_texture) * tv_loss_weight)
        total_loss = adv_loss + tv_loss

        optimizer.zero_grad()
        total_loss.backward()
        optimizer.step()

        adv_texture = model()
        dataset.set_textures(adv_texture)

        writer.add_scalar('Train/Loss/adv', adv_loss.item(), global_step)
        writer.add_scalar('Train/Loss/tv', tv_loss.item(), global_step)
        writer.add_scalar('Train/Loss/total', total_loss.item(), global_step)
        writer.add_scalar('Train/Metric/Ra', Ra.item(), global_step)
        writer.add_scalar('Train/Metric/Ed', Ed.item(), global_step)

        if i % 50 == 0:
            camouflage_scene_grid = make_grid(camouflage_scene[:8], nrow=4, padding=0)
            camouflage_scene_grid = Image.fromarray((camouflage_scene_grid.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))

            os.makedirs(os.path.join(exp_directory, 'image'), exist_ok=True)
            camouflage_scene_grid.save(
                os.path.join(exp_directory, 'image', '{}_{}_camouflage.jpg'.format(epoch, i // 50)))

            original_scene_grid = make_grid(original_scene[:8], nrow=4, padding=0)
            original_scene_grid = Image.fromarray((original_scene_grid.permute(1, 2, 0).detach().cpu().numpy() * 255).astype(np.uint8))

            os.makedirs(os.path.join(exp_directory, 'image'), exist_ok=True)
            original_scene_grid.save(
                os.path.join(exp_directory, 'image', '{}_{}_original.jpg'.format(epoch, i // 50)))

            with torch.no_grad():
                original_image = disp_to_image(original_disp)
                camouflage_image = disp_to_image(camouflage_disp)

            os.makedirs(os.path.join(exp_directory, 'depth'), exist_ok=True)
            original_image.save(
                os.path.join(exp_directory, 'depth', '{}_{}_original_disp.jpg'.format(epoch, i // 50)))
            camouflage_image.save(
                os.path.join(exp_directory, 'depth', '{}_{}_camouflage_disp.jpg'.format(epoch, i // 50)))

            save_texture(os.path.join(exp_directory, 'texture'), name='{}_{}'.format(epoch, i // 50))
        global_step += 1

    writer.flush()
    return global_step


def run_experiment(config):
    """
    运行单个实验，初始化数据集、模型和优化器，并执行训练和验证循环。

    参数:
        config (dict): 实验配置字典，包含以下键：
            - exp_name (str): 实验名称。
            - seed (int): 随机种子。
            - output_directory (str): 结果保存目录。
            - train_split (float): 训练集划分比例。
            - train_batch_size (int): 训练批次大小。
            - val_batch_size (int): 验证批次大小。
            - lr (float): 学习率。
            - eta_min (float): 学习率调度器的最小学习率。
            - epochs (int): 训练轮次。
            - dataset_params (dict): 数据集参数，包含 obj_name, uv_mask_name, data_dir, position_dir, apply_phys_aug, apply_img_aug, img_size。
            - mde_model_params (dict): 深度估计模型参数，包含 model_name, backbone, resolution。
            - adv_texture_params (dict): 对抗纹理生成器参数，包含 shape。
    """
    global dataset, model, model_name, optimizer, input_resize_mde, input_resize_image, input_resize_uv
    set_seed(config['seed'])
    device = torch.device("cuda:0") if torch.cuda.is_available() else torch.device("cpu")
    if not torch.cuda.is_available():
        print("WARNING: CPU only, this will be slow!")

    exp_directory = os.path.join(config['output_directory'], f"{config['exp_name']}_{datetime.now().strftime('%y%m%d_%H%M')}")
    os.makedirs(exp_directory, exist_ok=True)

    monodepth2, feed_height, feed_width = load_mde_model(**config['mde_model_params'], device=device)
    models = [monodepth2.eval()]

    dataset = CarlaDataset(device=device, **config['dataset_params'])

    train_size = int(len(dataset) * config['train_split'])
    val_size = len(dataset) - train_size
    train_dataset, val_dataset = random_split(dataset, [train_size, val_size])

    train_loader = DataLoader(
        dataset=train_dataset,
        batch_size=config['train_batch_size'],
        shuffle=True,
    )

    val_loader = DataLoader(
        dataset=val_dataset,
        batch_size=config['val_batch_size'],
        shuffle=True,
    )

    input_resize_mde = transforms.Resize([feed_height, feed_width])
    input_resize_image = transforms.Resize([dataset.image_size_y, dataset.image_size_x])
    input_resize_uv = transforms.Resize([dataset.binary_uv_mask.shape[0], dataset.binary_uv_mask.shape[1]])

    for param in monodepth2.parameters():
        param.requires_grad = False

    writer = SummaryWriter(log_dir=os.path.join(exp_directory, 'logs'))
    tensorboard_process = start_tensorboard(log_dir=os.path.join(exp_directory, 'logs'))

    if config['dataset_params']['apply_FBSG']:
        model = AdversarialTextureGenerator(**config['adv_texture_params']).to(device)
    else:
        class RandomTextureModel(torch.nn.Module):
            def __init__(self, height=256, width=256):
                super(RandomTextureModel, self).__init__()
                self.texture = torch.nn.Parameter(torch.randn(1, 3, height, width))

            def forward(self):
                return torch.sigmoid(self.texture)
        model = RandomTextureModel().to(device)

    model_name = config['mde_model_params']['model_name']
    optimizer = optim.Adam(model.parameters(), lr=config['lr'])
    lr_scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=config['epochs'], eta_min=config['eta_min'])
    global_step = 0

    adv_texture = model()
    dataset.set_textures(adv_texture)

    try:
        for epoch in range(1, config['epochs'] + 1):
            tqdm_train_loader = tqdm.tqdm(train_loader, desc=f"Epoch {epoch}/{config['epochs']}")
            global_step = train(tqdm_train_loader, models, writer, epoch, global_step, exp_directory, config['tv_loss_weight'], device)

            tqdm_val_loader = tqdm.tqdm(val_loader, desc=f"Validation {epoch}/{config['epochs']}")
            validation(tqdm_val_loader, models, writer, epoch, val_size, config['tv_loss_weight'], device)
            lr_scheduler.step()

    except KeyboardInterrupt:
        tensorboard_process.terminate()
    finally:
        tensorboard_process.terminate()


if __name__ == '__main__':
    base_config = {
        'seed': 42,
        'exp_name': 'test',
        'output_directory': 'results',
        'train_split': 0.8,
        'train_batch_size': 12,
        'val_batch_size': 8,
        'lr': 1e-4,
        'eta_min': 1e-6,
        'epochs': 15,
        'tv_loss_weight': 0.5,
        'dataset_params': {
            'obj_name': './Asset/lexus_hs.obj',
            'uv_mask_name': './Asset/lexus_mask.jpg',
            'data_dir': './dataset/ue4/rgb',
            'position_dir': './dataset/ue4',
            'apply_phys_aug': True,
            'apply_FBSG': True,
            'apply_trip_aug': True,
            'img_size': [1024, 320]
        },
        'mde_model_params': {
            'model_name': 'monodepth2',
            'backbone': 'resnet',
            'resolution': 'HR'
        },
        'adv_texture_params': {
            'shape': [256, 256],
            'num_freqs': 8

        }
    }
    # 定义变化的参数组合
    experiments = [
        {'exp_name': 'test'},
    ]

    for exp in experiments:
        config = update_new_config(base_config, exp)
        print(f"Running experiment: {config['exp_name']}")
        run_experiment(config)
