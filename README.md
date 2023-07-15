# Radar Enlighten the Dark: Enhancing Low-Visibility Perception for Automated Vehicles with Camera-Radar Fusion

<!-- PROJECT SHIELDS -->

[![python](https://img.shields.io/badge/-Python_3.6%2B-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.3%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![codecov](https://codecov.io/gh/PurdueDigitalTwin/REDFormer/branch/master/graph/badge.svg)](https://codecov.io/gh/PurdueDigitalTwin/REDFormer) \
[![MIT License](https://img.shields.io/badge/license-mit-darkred.svg)](https://github.com/PurdueDigitalTwin/REDFormer/blob/master/LICENSE)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/PurdueDigitalTwin/REDFormer/pulls)
[![contributors](https://img.shields.io/github/contributors/PurdueDigitalTwin/REDFormer.svg)](https://github.com/PurdueDigitalTwin/REDFormer/graphs/contributors)

<!-- PROJECT ILLUSTRATIONS -->

<br />
<div align="center">
    <p align="center">
        <img src="docs/img/overall.png", alt="arch", width="600"/>
    </p>
    <p align="center">
        <img src="docs/img/radar_backbone.png", alt="rad", width="600">
    </p>
</div>

______________________________________________________________________

## Table of Contents

- [Radar Enlighten the Dark: Enhancing Low-Visibility Perception for Automated Vehicles with Camera-Radar Fusion](#radar-enlighten-the-dark-enhancing-low-visibility-perception-for-automated-vehicles-with-camera-radar-fusion)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
      - [Download nuscenes full dataset](#download-nuscenes-full-dataset)
      - [Generating annotation files](#generating-annotation-files)
    - [Training](#training)
    - [Test](#test)
  - [Citatation](#citatation)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)

______________________________________________________________________

## About

In this work, we propose a novel transformer-based  3D object detection model \`\`REDFormer'' to tackle low visibility conditions, exploiting the power of a more practical and cost-effective solution by leveraging bird's-eye-view camera-radar fusion. Using the nuScenes dataset with multi-radar point clouds, weather information, and time-of-day data, our model outperforms state-of-the-art (SOTA) models on classification and detection accuracy. Finally, we provide extensive ablation studies of each model component on their contributions to address the above-mentioned challenges. Particularly, it is shown in the experiments that our model achieves a significant performance improvement over the baseline model in low-visibility scenarios, specifically exhibiting a **31.31%** increase in rainy scenes and a **46.99%** enhancement in nighttime scenes.

______________________________________________________________________

## Getting Started

### Installation

Please refer to our [installation guide](docs/installation.md) for details.

### Data Preparation

#### Download nuscenes full dataset

Please refer to nuScenes official website to download nuScenes v1.0 full dataset and CAN bus expansion. [Nuscenes Official Website](https://www.nuscenes.org/download)

#### Generating annotation files

```
bash scripts/create_data.sh
```

**Download the checkpoint files**

Please put the 'bevformer_raw.pth' to 'ckpts/raw_model' and put 'R101-DCN' to folder 'ckpts'.

|   Backbone    |                                                       Download                                                       |
| :-----------: | :------------------------------------------------------------------------------------------------------------------: |
|   R101-DCN    | [model download](https://github.com/cancui19/model_storage/releases/download/redformer/r101_dcn_fcos3d_pretrain.pth) |
| bevformer_raw |      [model download](https://github.com/cancui19/model_storage/releases/download/redformer/bevformer_raw.pth)       |

|     Model     |                                               Download                                                |
| :-----------: | :---------------------------------------------------------------------------------------------------: |
| Our REDFormer | [model download](https://github.com/cancui19/model_storage/releases/download/redformer/redformer.pth) |

**Folder structure**

```plain
REDFormer
├── ckpts        # folder for checkpoints
│   ├── raw_model/
│   │   └── bevformer_raw.pth
│   ├── r101_dcn_fcos3d_pretrain.pth
│   └── redformer.pth
├── data         # folder for NuScenes dataset
│   ├── nuscenes/
│   │   ├── full/
│   │   │   ├── can_bus/
│   │   │   ├── maps/
│   │   │   ├── samples/
│   │   │   ├── sweeps/
│   │   │   ├── v1.0-test/
│   │   │   ├── v1.0-trainval/
│   │   │   ├── nuscenes_infos_ext_train.pkl
│   │   │   ├── nuscenes_infos_ext_val.pkl
│   │   │   ├── nuscenes_infos_ext_rain_val.pkl
│   │   │   └── nuscenes_infos_ext_night_val.pkl
├── environment.yml
├── LICENSE
├── README.md
├── scripts
├── setup.py
└── tools
```

### Training

```
bash scripts/train.sh
```

### Test

```
bash scripts/test.sh
```

If you want to test the performance on rain or night scenes, please go the config file [Here](projects/configs/redformer/redformer.py) (`projects/configs/redformer/redformer.py`)  and modify the value of `environment_test_subset`.

______________________________________________________________________

## Citatation

If you find REDFormer usefule, you are highly encouraged to cite our paper:

```
@misc{
  cui2023radar,
  title={Radar Enlighten the Dark: Enhancing Low-Visibility Perception for Automated Vehicles with Camera-Radar Fusion},
  author={Can Cui and Yunsheng Ma and Juanwu Lu and Ziran Wang},
  year={2023},
  eprint={2305.17318},
  archivePrefix={arXiv},
  primaryClass={cs.CV}
}
```

______________________________________________________________________

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

______________________________________________________________________

## Acknowledgement

We attribute our work to the following inspiring open source projects:

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [mmdetetection3d](https://github.com/open-mmlab/mmdetection3d)
