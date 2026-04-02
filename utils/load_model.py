import sys
import os
import torch
from torch import nn
project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def load_mde_model(model_name, resolution=None, backbone=None, device='cuda'):
    from models.utils import DepthModelWrapper
    if model_name == 'monodepth2':
        sys.path.append(os.path.join(project_root, 'third_party', 'mde', 'monodepth2'))
        import networks as net_m2
        if resolution == 'HR':
            model_path = os.path.join("third_party", "mde", "monodepth2", "models", "mono+stereo_1024x320")
        elif resolution == 'MR':
            model_path = os.path.join("third_party", "mde", "monodepth2", "models", "mono+stereo_640x192")
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

    elif model_name == 'depthhints':
        sys.path.append(os.path.join(project_root, 'third_party', 'mde', 'depthhints'))
        import networks as net_dh

        if resolution == 'HR':
            model_path = os.path.join("third_party", "mde", "depthhints", "models")
        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

        print("-> Loading model from ", model_path)
        encoder_path = os.path.join(model_path, "encoder.pth")
        depth_decoder_path = os.path.join(model_path, "depth.pth")

        depth_encoder = net_dh.ResnetEncoder(18, False)
        depth_decoder = net_dh.DepthDecoder(num_ch_enc=depth_encoder.num_ch_enc, scales=range(4))

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

    elif model_name == 'monovit':
        sys.path.append(os.path.join(project_root, 'third_party', 'mde', 'MonoViT'))
        sys.path.append(os.path.join(project_root, 'third_party', 'mmcv'))
        import networks as net_mv

        if resolution == 'HR':
            depth_path = os.path.join("third_party", "mde", "MonoViT", "models", "MonoViT_MS_1024x320")
            depth_path = os.path.join(depth_path, "depth.pth")

            depth_dict = torch.load(depth_path)
            feed_height = depth_dict['height']
            feed_width = depth_dict['width']

            new_dict = {}
            for k, v in depth_dict.items():
                name = k[7:]
                new_dict[name] = v

            vit_model = net_mv.DeepNet('mpvitnet')
            vit_model.load_state_dict({k: v for k, v in new_dict.items() if k in vit_model.state_dict()})
            depth_model = DepthModelWrapper(model=vit_model).to(device)

            for param in depth_model.parameters():
                param.requires_grad = False

            return depth_model, feed_height, feed_width

        elif resolution == 'MR':
            model_path = os.path.join("third_party", "mde", "MonoViT", "models", "MonoViT_MS_640x192")
            encoder_path = os.path.join(model_path, "encoder.pth")
            decoder_path = os.path.join(model_path, "depth.pth")

            encoder = net_mv.mpvit_small()
            encoder.num_ch_enc = [64, 128, 216, 288, 288]

            encoder_dict = torch.load(encoder_path)
            feed_height = encoder_dict['height']
            feed_width = encoder_dict['width']

            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})

            depth_decoder = net_mv.DepthDecoder()
            depth_decoder.load_state_dict(torch.load(decoder_path))

            depth_model = DepthModelWrapper(encoder, depth_decoder).to(device)
            depth_model.cuda()

            for param in depth_model.parameters():
                param.requires_grad = False

            return depth_model, feed_height, feed_width

        else:
            raise ValueError(f"Unsupported resolution: {resolution}")

    elif model_name == 'robustdepth':
        sys.path.append(os.path.join(project_root, 'third_party', 'mde', 'robustdepth'))
        import Robust_Depth.networks as net_rd
        import Robust_Depth.networksvit as netvit_rd

        if backbone == 'vit':
            model_path = os.path.join("third_party", "mde", "robustdepth", "models", "ViT")
            encoder_path = os.path.join(model_path, "encoder.pth")
            decoder_path = os.path.join(model_path, "depth.pth")
            encoder_dict = torch.load(encoder_path, map_location='cuda:0')
            feed_height = encoder_dict['height']
            feed_width = encoder_dict['width']
            encoder = netvit_rd.mpvit_small()
            encoder.num_ch_enc = [64, 128, 216, 288, 288]
            depth_decoder = netvit_rd.DepthDecoder()

            model_dict = encoder.state_dict()
            encoder.load_state_dict({k: v for k, v in encoder_dict.items() if k in model_dict})
            depth_decoder.load_state_dict(torch.load(decoder_path, map_location='cuda:0'))
            depth_model = DepthModelWrapper(encoder, depth_decoder).to(device)

            for param in depth_model.parameters():
                param.requires_grad = False

            return depth_model, feed_height, feed_width

        elif backbone == 'resnet':
            model_path = os.path.join("third_party", "mde", "robustdepth", "models", "Resnet")
            encoder_path = os.path.join(model_path, "encoder.pth")
            decoder_path = os.path.join(model_path, "depth.pth")

            encoder = net_rd.ResnetEncoder(18, False)
            loaded_dict_enc = torch.load(encoder_path, map_location="cuda:0")
            feed_height = loaded_dict_enc['height']
            feed_width = loaded_dict_enc['width']
            filtered_dict_enc = {k: v for k, v in loaded_dict_enc.items() if k in encoder.state_dict()}
            encoder.load_state_dict(filtered_dict_enc)

            print("   Loading pretrained decoder")
            depth_decoder = net_rd.DepthDecoder(num_ch_enc=encoder.num_ch_enc, scales=range(4))

            loaded_dict = torch.load(decoder_path, map_location="cuda:0")
            depth_decoder.load_state_dict(loaded_dict)

            depth_model = DepthModelWrapper(encoder, depth_decoder).to(device)
            for param in depth_model.parameters():
                param.requires_grad = False

            return depth_model, feed_height, feed_width

        else:
            raise ValueError(f"Unsupported backbone: {backbone}")

    else:
        raise ValueError(f"Unsupported MDE model: {model_name}")


def load_detection_model(model_name, device='cuda'):
    if model_name == 'yolo11':
        from ultralytics import YOLO
        model = YOLO(os.path.join(project_root, 'checkpoints/yolo11n.pt'))
        return model

    elif model_name == 'dfine':
        sys.path.append(os.path.join(project_root, 'third_party', 'detection', 'D-FINE'))
        from src.core import YAMLConfig
        cfg = os.path.join(project_root, r'third_party/detection/D-FINE/dfine_hgnetv2_n_coco.yml')
        resume = os.path.join(project_root, r'third_party/detection/D-FINE/dfine_n_coco.pth')

        cfg = YAMLConfig(cfg, resume=resume)

        if "HGNetv2" in cfg.yaml_cfg:
            cfg.yaml_cfg["HGNetv2"]["pretrained"] = False

        if resume:
            checkpoint = torch.load(resume)
            if "ema" in checkpoint:
                state = checkpoint["ema"]["module"]
            else:
                state = checkpoint["model"]
        else:
            raise AttributeError("Only support resume to load model.state_dict by now.")

        # Load train mode state and convert to deploy mode
        cfg.model.load_state_dict(state)

        class Model(nn.Module):
            def __init__(self):
                super().__init__()
                self.model = cfg.model.deploy()
                self.postprocessor = cfg.postprocessor.deploy()

            def forward(self, images, orig_target_sizes):
                outputs = self.model(images)
                outputs = self.postprocessor(outputs, orig_target_sizes)
                return outputs

        model = Model().to(device)
        return model

    else:
        raise ValueError(f"Unsupported detection model: {model_name}")


def load_segmentation_model(model_name, dataset_name, device='cuda'):
    if 'cityscapes' in dataset_name:
        num_classes = 19
    elif 'coco' in dataset_name:
        num_classes = 182
    elif 'ade20k' in dataset_name:
        num_classes = 150
    else:
        raise ValueError(f'Unsupported dataset: {dataset_name}')

    if model_name == 'deeplabv3plus':
        sys.path.append(os.path.join(project_root, 'third_party', 'semseg', 'DeepLabV3Plus-Pytorch'))
        from network import modeling

        model = modeling.deeplabv3plus_mobilenet(num_classes=num_classes).to(device)
        model.load_state_dict(torch.load(os.path.join(project_root, 'checkpoints/best_deeplabv3plus_mobilenet_cityscapes_os16.pth'))['model_state'])
        model = model.eval()
        return model

    elif model_name == 'pidnet':
        sys.path.insert(0, os.path.join(project_root, 'third_party', 'semseg', 'PIDNet'))
        from models.pidnet import get_pred_model

        model = get_pred_model(name="pidnet-l", num_classes=num_classes).to(device)
        pretrained_dict = torch.load(
            os.path.join(project_root, 'third_party/semseg/PIDNet/pretrained_models/cityscapes/PIDNet_L_Cityscapes_test.pt'))

        if 'state_dict' in pretrained_dict:
            pretrained_dict = pretrained_dict['state_dict']
        model_dict = model.state_dict()
        pretrained_dict = {k[6:]: v for k, v in pretrained_dict.items() if (k[6:] in model_dict and v.shape == model_dict[k[6:]].shape)}
        model_dict.update(pretrained_dict)
        model.load_state_dict(model_dict)
        model.eval()
        return model

    elif model_name == 'sctnet':
        sys.path.append(os.path.join(project_root, 'third_party', 'mmcv'))
        sys.path.insert(0, os.path.join(project_root, 'third_party', 'semseg', 'SCTNet'))
        import mmcv
        from mmcv.runner import load_checkpoint
        from mmseg.models import build_segmentor

        if 'cityscapes' in dataset_name:
            cfg = mmcv.Config.fromfile(
                os.path.join(project_root, "third_party/semseg/SCTNet/configs/sctnet/cityscapes/sctnet-s_seg50_8x2_160k_cityscapes.py"))

            cfg.model.pretrained = None
            cfg.model.backbone.init_cfg['type'] = 'None'
            cfg.model.auxiliary_head[1].init_cfg['type'] = 'None'
            model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')).to(device)
            model.eval()
            model.cfg = cfg
            load_checkpoint(model, os.path.join(project_root, "checkpoints/SCTNet-S-Seg50.pth"))
            return model

        elif 'ade20k' in dataset_name:
            cfg = mmcv.Config.fromfile(os.path.join(project_root, "third_party/semseg/SCTNet/configs/sctnet/ADE20K/sctnet-b_8x4_160k_ade.py"))
            cfg.model.pretrained = None
            cfg.model.backbone.init_cfg['type'] = 'None'
            cfg.model.auxiliary_head[1].init_cfg['type'] = 'None'
            model = build_segmentor(cfg.model, test_cfg=cfg.get('test_cfg')).to(device)
            model.eval()
            model.cfg = cfg
            load_checkpoint(model, os.path.join(project_root, "checkpoints/SCTNet-B-ADE20K.pth"))
            return model

    else:
        raise ValueError(f"Unsupported segmentation model: {model_name}")
