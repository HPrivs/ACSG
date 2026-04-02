import os
import tqdm
from PIL import Image, ImageDraw, ImageFont

import torchmetrics
import torch
from torchvision.transforms import transforms
from torch.utils.data import DataLoader

from data.dataset import DetectionImageDataset
from utils.load_model import load_detection_model
import numpy as np

def center_box_detected(cls, xyxy, conf, W, H):
    cx, cy = W / 2, H / 2
    for i in range(len(xyxy)):
        if cls[i] == 2:
            x1, y1, x2, y2 = xyxy[i]
            if x1 <= cx <= x2 and y1 <= cy <= y2:
                return cls[i].unsqueeze(0).cpu().to(torch.int64), xyxy[i].unsqueeze(0).cpu(), conf[i].unsqueeze(0).cpu()
    return torch.empty((0,), dtype=torch.int64), torch.empty((0, 4), dtype=torch.float32), torch.empty((0,), dtype=torch.float32)


def draw_single(image: Image.Image, boxes, scores, thrh=0.4, save_path=None):
    draw = ImageDraw.Draw(image)

    def is_dark_color(color):
        r, g, b = color
        return (0.299 * r + 0.587 * g + 0.114 * b) < 130

    try:
        font = ImageFont.truetype("arial.ttf", size=20)
    except:
        font = ImageFont.load_default()

    LABEL_NAME = "Car"
    COLOR = (30, 144, 255)

    mask = scores > thrh
    boxes = boxes[mask].tolist()
    scores = scores[mask].tolist()

    for i, (x1, y1, x2, y2) in enumerate(boxes):
        score = scores[i]
        label_text = f"{LABEL_NAME} {score:.2f}"

        left, top, right, bottom = font.getbbox(label_text)
        text_w = right - left
        text_h = bottom - top

        pad_x, pad_y = 6, 4

        bg_left = x1
        bg_top = y1 - (text_h + 2 * pad_y)
        bg_right = x1 + text_w + 2 * pad_x
        bg_bottom = y1

        if bg_top < 0:
            bg_top = y1
            bg_bottom = y1 + text_h + 2 * pad_y

        text_color = "white" if is_dark_color(COLOR) else "black"
        draw.rectangle([bg_left, bg_top, bg_right, bg_bottom], fill=COLOR)
        draw.text((bg_left + pad_x, bg_top + pad_y), label_text, fill=text_color, font=font)
        draw.rectangle([x1, y1, x2, y2], outline=COLOR, width=4)

    if save_path:
        image.save(save_path)


def yolo_metric_to_dict(result, gt_bbox):
    H, W = result.orig_shape
    cls, xyxy, conf = center_box_detected(result.boxes.cls, result.boxes.xyxy, result.boxes.conf, W, H)

    pred_dict = {
        "boxes": xyxy,
        "scores": conf,
        "labels": cls
    }

    target_dict = {
        "boxes": gt_bbox,
        "labels": torch.tensor([2], dtype=torch.int64)
    }
    return pred_dict, target_dict


def dfine_inference_single(model, image_path, gt_bbox, conf_thres=0.3):
    im_pil = Image.open(image_path).convert("RGB")
    w, h = im_pil.size
    orig_size = torch.tensor([[w, h]]).to('cuda')

    resizer = transforms.Compose([
        transforms.Resize((640, 640)),
        transforms.ToTensor(),
    ])
    im_data = resizer(im_pil).unsqueeze(0).to('cuda')

    with torch.no_grad():
        labels, boxes, scores = model(im_data, orig_size)

    lab = labels[scores > conf_thres]
    box = boxes[scores > conf_thres]
    scrs = scores[scores > conf_thres]

    cls, xyxy, conf = center_box_detected(lab, box, scrs, w, h)

    pred_dict = {
        "boxes": xyxy,
        "scores": conf,
        "labels": cls
    }

    target_dict = {
        "boxes": gt_bbox,
        "labels": torch.tensor([2], dtype=torch.int64)
    }

    return pred_dict, target_dict, im_pil

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

def print_model_layers(model, prefix=''):
    for name, module in model.named_children():
        full_name = f"{prefix}.{name}" if prefix else name
        print(full_name, type(module))
        if len(list(module.children())) > 0:
            print_model_layers(module, prefix=full_name)

def register_hooks_dfine(model):
    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for name, module in model.named_modules():
        if name.startswith("model.backbone.stages") and name.count('.') == 3:
            module.register_forward_hook(hook_fn(name))
            print(f"[HOOK] STAGE: {name}")

    return model

def register_hooks_yolo11(model):
    FEATURE_CLASSES = {
        'Conv', 'C3k2', 'C3k', 'SPPF', 'C2PSA', 'PSABlock'
    }

    if isinstance(model, torch.nn.DataParallel):
        model = model.module

    for name, module in model.named_modules():
        if name.startswith("model.model."):
            try:
                idx = int(name.split(".")[2])
            except:
                continue

            if type(module).__name__ in FEATURE_CLASSES:
                module.register_forward_hook(hook_fn(name))
                print(f"[HOOK] {type(module).__name__}: {name}")

    return model

def save_features(features, model_name, exp_name, img_name):
    save_base = os.path.join(f'./simulation/detection/{model_name}/hook/{exp_name}/{img_name}')
    os.makedirs(save_base, exist_ok=True)

    for feat_name, feat in features.items():
        arr = feat[0].numpy()
        arr = arr - arr.min()
        arr = arr / (arr.max() + 1e-8)
        arr = (arr * 255).astype(np.uint8)
        file_name = f"{feat_name.replace('.', '_')}.png"
        Image.fromarray(arr).save(os.path.join(save_base, file_name))

if __name__ == "__main__":
    dataset = DetectionImageDataset(my_dir=r'./simulation/original/auditt', parms_dir='./simulation/original/position_params.json',
                                    obj_dir='./Asset/auditt.obj', img_size=(1024, 320))
    dataloader = DataLoader(dataset, batch_size=1, shuffle=True)
    tqdm_loader = tqdm.tqdm(dataloader, desc=f"Simulation Test")
    metric = torchmetrics.detection.MeanAveragePrecision(iou_type='bbox')

    model_name = 'yolo11'
    exp_name = os.path.basename(os.path.dirname(dataset.my_dir)) + "-" + os.path.basename(dataset.my_dir)
    model = load_detection_model(model_name, device='cuda')

    model = register_hooks_yolo11(model)

    for path, name, gt_bbox in tqdm_loader:
        if model_name == 'yolo11':
            results = model(path, conf=0.3)
            for i, result in enumerate(results):
                pred_dict, target_dict = yolo_metric_to_dict(result, gt_bbox[i])
                metric.update([pred_dict], [target_dict])
                save_path = os.path.join(f'./simulation/detection/{model_name}/{exp_name}/' + name[i] + '.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                result.save(filename=save_path)
                save_features(features, model_name, exp_name, name[i])

        elif model_name == 'dfine':
            for i in range(len(path)):
                pred_dict, target_dict, im_pil = dfine_inference_single(model, path[i], gt_bbox[i], conf_thres=0.3)
                metric.update([pred_dict], [target_dict])
                save_path = os.path.join(f'./simulation/detection/{model_name}/{exp_name}/' + name[i] + '.png')
                os.makedirs(os.path.dirname(save_path), exist_ok=True)
                draw_single(image=im_pil, boxes=pred_dict["boxes"], scores=pred_dict["scores"], thrh=0.3, save_path=save_path)
                save_features(features, model_name, exp_name, name[i])

    metrics = metric.compute()
    print(metrics)
