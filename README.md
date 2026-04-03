<h2 align="center">ACSG: Structure-Guided Physical Adversarial Camouflage for Monocular Depth Estimation</h2>

<p align="center">
  <img src="https://img.shields.io/github/stars/HPrivs/ACSG?style=social" alt="GitHub stars" />
  <img alt="Static Badge" src="https://img.shields.io/badge/license-Apache%202.0-blue.svg" />
</p>

<p align="center">
  Official PyTorch implementation of <b>"ACSG: Adversarial Camouflage with Structural Guidance for Monocular Depth Estimation"</b>
</p>

## Abstract
Monocular depth estimation (MDE) is widely used in autonomous driving, yet is vulnerable to real-world physical adversarial perturbations. However, physical adversarial camouflage against MDE remains largely underexplored. Existing methods, which typically optimize textures in UV space or from random initialization, exhibit limited effectiveness in perturbing depth predictions due to the lack of intrinsic spectral and topological structural guidance. To address these limitations, we propose Adversarial Camouflage With Structural Guidance (ACSG). Designed to disrupt the geometric cues relied upon by MDE, ACSG leverages toroidal Fourier positional encoding to generate multi-scale structured adversarial textures with topological continuity. These textures are applied via a Hybrid UV-Triplanar Mapping scheme to enable geometry-agnostic deployment while strictly preserving non-modifiable regions. Additionally, a Physical Effect Simulation module is integrated to ensure robustness against real-world environmental dynamics. Extensive simulation experiments demonstrate that ACSG outperforms existing methods across diverse weather conditions, vehicle and obstacle types, camera viewpoints, and perception tasks. Furthermore, real-world experiments confirm its physical realizability, inducing average depth deviations exceeding 20m indoors and 15m outdoors.

## Framework
![image-framework](https://raw.githubusercontent.com/HPrivs/ACSG/main/framework.jpg)

## Installation
```bash
conda create -n acsg python=3.9
conda activate acsg

git clone https://github.com/HPrivs/ACSG.git
cd ACSG
pip install -r requirements.txt
```
## File Structure
```text
ACSG/
├── Asset/               # 3D assets, including meshes and UV texture materials
├── data/                # Data loading pipelines and PyTorch3D-based differentiable rendering
├── dataset/             # Prepared datasets
├── models/              # Adversarial texture generation models
├── third_party/         # External repositories (e.g., Monodepth2)
├── utils/               # Utility scripts, including CARLA data generation
├── train.py             # Main script for adversarial camouflage training
└── test.py              # Evaluation scripts
```

## Preparation

### 1. Dataset Extraction
Download the datasets and extract them into the `datasets/` folder. The structure should be:
```text
ACSG/
└── dataset/
    └── ue4/
        ├── background/      # Background scenes without vehicles
        ├── rgb/             # Images containing target vehicles
        └── position.json    # Camera pose metadata
```

### 2. Pre-trained MDE Models
Download the official **MonoDepth2** model and place the weights as shown below:
```text
ACSG/
└── third_party/
    └── mde/
        └── monodepth2/
            └── models/
                └── mono+stereo_1024x320/  # Place .pth weights here
```

## Quick Start
```bash
# Run the training script
python train.py
```

## Dataset Links
* [**QuarkNetdisk**](https://pan.quark.cn/s/5fb62d854152)
* [**GoogleDrive**](https://drive.google.com/file/d/1O-NUCnc64N69Qnj_gsh-asWEE1zYNX-f/view?usp=sharing)

## Acknowledgements
* **3D2Fool** - [Paper](http://arxiv.org/abs/2403.17301) | [Source Code](https://github.com/Gandolfczjh/3D2Fool)