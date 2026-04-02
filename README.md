<h2  align="center">ACSG: Structure-Guided Physical Adversarial Camouflage for Monocular Depth Estimation</h2>
<p align="center">
  <img src="https://img.shields.io/github/stars/HPrivs/ACSG?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
</p>

Official PyTorch implementation of **"ACSG: Adversarial Camouflage with Structural Guidance for Monocular Depth Estimation"**


## Abstract
Monocular depth estimation (MDE) is widely used in autonomous driving, yet is vulnerable to real-world physical adversarial perturbations. However, physical adversarial camouflage against MDE remains largely underexplored. Existing methods, which typically optimize textures in UV space or from random initialization, exhibit limited effectiveness in perturbing depth predictions due to the lack of intrinsic spectral and topological structural guidance. To address these limitations, we propose Adversarial Camouflage With Structural Guidance (ACSG). Designed to disrupt the geometric cues relied upon by MDE, ACSG leverages toroidal Fourier positional encoding to generate multi-scale structured adversarial textures with topological continuity. These textures are applied via a Hybrid UV-Triplanar Mapping scheme to enable geometry-agnostic deployment while strictly preserving non-modifiable regions. Additionally, a Physical Effect Simulation module is integrated to ensure robustness against real-world environmental dynamics. Extensive simulation experiments demonstrate that ACSG outperforms existing methods across diverse weather conditions, vehicle and obstacle types, camera viewpoints, and perception tasks. Furthermore, real-world experiments confirm its physical realizability, inducing average depth deviations exceeding 20m indoors and 15m outdoors.
## Framework
![image-framework](https://github.com/Gandolfczjh/3D2Fool/blob/main/framework.jpg)

## Installation
```bash
conda create -n acsg python=3.9
conda activate acsg
pip install -r requirements.txt

## 
python train.py
```
* data_loader_mde.py
  > class MyDataset: load training set
  > + data_dir: rgb background images path
  > + obj_name: car model path
  > + camou_mask: mask path (the texture area to attack)
  > + tex_trans_flag: TC flag
  > + phy_trans_flag: PA flag
  > + self.set_textures(self, camou): camou is texture seed
  > + camera_pos: camera relative position in carla
* attack_base.py
  > + camou_mask: camouflage texture mask path
  > + camou_shape: shape of camouflage texture
  > + obj_name: car model path
  > + train_dir: rgb background images
  > + log_dir: result save path

## Dataset
* [QuarkNetdisk](https://pan.quark.cn/s/5fb62d854152)
* [GoogleDrive](google drive)
  > + ./rgb/*.jpg: background images
  > + ./ann.pkl: camera position matrix

## Acknowledgements
* 3D2Fool - [**Paper**](http://arxiv.org/abs/2403.17301)
| [Source Code](https://github.com/Gandolfczjh/3D2Fool)
