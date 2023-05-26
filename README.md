<div align="left">

# Radar Enlighten the Dark: Enhancing Low-Visibility Perception for Automated Vehicles with Camera-Radar Fusion

</div>

<!-- PROJECT SHIELDS -->
<div align="center">

[![python](https://img.shields.io/badge/-Python_3.6%2B-blue?logo=python&logoColor=white)](https://github.com/pre-commit/pre-commit)
[![pytorch](https://img.shields.io/badge/PyTorch_1.3%2B-ee4c2c?logo=pytorch&logoColor=white)](https://pytorch.org/get-started/locally/)
[![codecov](https://codecov.io/gh/PurdueDigitalTwin/REDFormer/branch/master/graph/badge.svg)](https://codecov.io/gh/PurdueDigitalTwin/REDFormer) <br>
[![MIT License](https://img.shields.io/badge/license-mit-darkred.svg)](https://github.com/PurdueDigitalTwin/REDFormer/blob/master/LICENSE)
[![PRs](https://img.shields.io/badge/PRs-welcome-brightgreen.svg)](https://github.com/PurdueDigitalTwin/REDFormer/pulls)
[![contributors](https://img.shields.io/github/contributors/PurdueDigitalTwin/REDFormer.svg)](https://github.com/PurdueDigitalTwin/REDFormer/graphs/contributors)

</div>

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

---

## Table of Contents

- [Radar Enlighten the Dark: Enhancing Low-Visibility Perception for Automated Vehicles with Camera-Radar Fusion](#radar-enlighten-the-dark-enhancing-low-visibility-perception-for-automated-vehicles-with-camera-radar-fusion)
  - [Table of Contents](#table-of-contents)
  - [About](#about)
  - [Getting Started](#getting-started)
    - [Installation](#installation)
    - [Data Preparation](#data-preparation)
    - [Training](#training)
    - [Test](#test)
  - [Citatation](#citatation)
  - [License](#license)
  - [Acknowledgement](#acknowledgement)

---

## About

In this work, we propose a novel transformer-based  3D object detection model ``REDFormer'' to tackle low visibility conditions, exploiting the power of a more practical and cost-effective solution by leveraging bird's-eye-view camera-radar fusion. Using the nuScenes dataset with multi-radar point clouds, weather information, and time-of-day data, our model outperforms state-of-the-art (SOTA) models on classification and detection accuracy. Finally, we provide extensive ablation studies of each model component on their contributions to address the above-mentioned challenges. Particularly, it is shown in the experiments that our model achieves a significant performance improvement over the baseline model in low-visibility scenarios, specifically exhibiting a **31.31%** increase in rainy scenes and a **46.99%** enhancement in nighttime scenes.

---

## Getting Started

### Installation

Please refer to our [installation guide](docs/installation.md) for details.

### Data Preparation

### Training

### Test

---

## Citatation

If you find REDFormer usefule, you are highly encouraged to cite our paper:

```bibtex
```

---

## License

Distributed under the MIT License. See [`LICENSE`](LICENSE) for more information.

---

## Acknowledgement

We attribute our work to the following inspiring open source projects:

- [BEVFormer](https://github.com/fundamentalvision/BEVFormer)
- [mmdetetection3d](https://github.com/open-mmlab/mmdetection3d)
