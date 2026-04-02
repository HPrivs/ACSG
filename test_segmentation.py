import os
import sys
from PIL import Image
import tqdm

import numpy as np
import torch
from torchvision import transforms
from torch.utils.data import DataLoader
from torch.nn.functional import interpolate

from utils.load_model import load_segmentation_model
from data.dataset import SegmentationImageDataset

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'mmcv')))
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), 'third_party', 'semseg', 'SCTNet')))
from mmseg.apis import inference_segmentor
from mmseg.datasets import ADE20KDataset, COCOStuffDataset, CityscapesDataset

def inference_single(model_name, dataset_name, model, path, mask, device='cuda'):
    if 'cityscapes' in dataset_name:
        car_id = 13
    elif 'coco' in dataset_name:
        car_id = 2
    elif 'ade20k' in dataset_name:
        car_id = 20
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    img_pil = Image.open(path).convert("RGB")
    size = mask.shape[-2:]
    if model_name == 'deeplabv3plus':
        processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = processor(img_pil).unsqueeze(0).to(device)
        output = model(img_tensor)
        pred = output.max(1)[1].squeeze(0).cpu().numpy()  # HW
        pred_mask = (pred == car_id).astype(np.uint8)
        gt_mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        inter_sum = np.sum((pred_mask == 1) & (gt_mask == 1))
        gt_sum = np.sum(gt_mask == 1)

    elif model_name == 'pidnet':
        processor = transforms.Compose([
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        img_tensor = processor(img_pil).unsqueeze(0).to(device)
        output = model(img_tensor)
        pred = interpolate(output, size=size, mode='bilinear', align_corners=True)
        pred = torch.argmax(pred, dim=1).squeeze(0).cpu().numpy()
        pred_mask = (pred == car_id).astype(np.uint8)
        gt_mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        inter_sum = np.sum((pred_mask == 1) & (gt_mask == 1))
        gt_sum = np.sum(gt_mask == 1)

    elif model_name == 'sctnet':
        output = inference_segmentor(model, path)
        pred = output[0]
        pred_mask = (pred == car_id).astype(np.uint8)
        gt_mask = mask.squeeze(0).cpu().numpy().astype(np.uint8)

        inter_sum = np.sum((pred_mask == 1) & (gt_mask == 1))
        gt_sum = np.sum(gt_mask == 1)

    return pred, inter_sum, gt_sum


def draw_result(pred, model_name, exp_name, dataset_name):
    if 'cityscapes' in dataset_name:
        PALETTE = CityscapesDataset.PALETTE
    elif 'coco' in dataset_name:
        PALETTE = COCOStuffDataset.PALETTE
    elif 'ade20k' in dataset_name:
        PALETTE = ADE20KDataset.PALETTE
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    palette = np.array(PALETTE, dtype=np.uint8)
    h, w = pred.shape
    colorized_pred = np.zeros((h, w, 3), dtype=np.uint8)
    for idx, color in enumerate(palette):
        colorized_pred[pred == idx] = color

    colorized_pred = Image.fromarray(colorized_pred)
    save_path = os.path.join(f'./simulation/segmentation/{model_name}/{dataset_name}/{exp_name}/' + name[i] + '.png')
    os.makedirs(os.path.dirname(save_path), exist_ok=True)
    colorized_pred.save(save_path)

features = {}
def hook_fn(name):
    def hook(module, input, output):
        if isinstance(output, torch.Tensor):
            feat = output.detach()
            if feat.ndim == 4:
                feat = feat[0].mean(dim=0, keepdim=True)
            elif feat.ndim == 3:
                feat = feat.unsqueeze(0)
            features[name] = feat.cpu()
    return hook

def register_hooks_by_name(model, class_name_list=['CFBlock', 'Sequential']):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for name, module in model.named_modules():
        if type(module).__name__ in class_name_list:
            module.register_forward_hook(hook_fn(name))
            print(f"[HOOK] {type(module).__name__}: {name}")

    return model


def print_model_layers(model, prefix=''):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(full_name, type(module))
        if len(list(module.children())) > 0:
            print_model_layers(module, prefix=full_name)

def save_features(features, model_name, dataset_name, exp_name, img_name):
    save_base = f'./simulation/segmentation/{model_name}/{dataset_name}/hook/{exp_name}/{img_name}'
    os.makedirs(save_base, exist_ok=True)

    for feat_name, feat in features.items():
        arr = feat[0].numpy()
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        file_name = f"{feat_name.replace('.', '_')}.png"
        Image.fromarray(arr).save(os.path.join(save_base, file_name))

if __name__ == "__main__":
    dataset = SegmentationImageDataset(my_dir=r'./simulation/prefix/patch2camou_random', parms_dir='./simulation/original/position_params.json',
                                       obj_dir='./Asset/auditt.obj',
                                       img_size=(1024, 320))
    dataloader = DataLoader(dataset, batch_size=16, shuffle=False)
    tqdm_loader = tqdm.tqdm(dataloader, desc=f"Simulation Test")

    model_name = 'sctnet'
    dataset_name = 'ade20k'
    model = load_segmentation_model(model_name, dataset_name)
    model = register_hooks_by_name(model, ['CFBlock', 'Sequential'])

    # print_model_layers(model)
    exp_name = os.path.basename(os.path.dirname(dataset.my_dir)) + "-" + os.path.basename(dataset.my_dir)

    total_intersection = 0.0
    total_gt_count = 0.0
    for path, name, mask in tqdm_loader:
        for i in range(len(path)):
            pred, inter_sum, gt_sum = inference_single(model_name, dataset_name, model, path[i], mask[i])
            total_intersection += inter_sum
            total_gt_count += gt_sum

            # save feature
            save_features(features, model_name, dataset_name, exp_name, name[i])

            # draw
            draw_result(pred, model_name, exp_name, dataset_name)

    global_percentage = (total_intersection / total_gt_count) * 100
    print(f"Global Pixel Accuracy for car class: {global_percentage:.2f}%")