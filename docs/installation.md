# Installation

We built our model with the `mmdetection3d` project. Users may refer to their [`getting_started`](https://github.com/open-mmlab/mmdetection3d/blob/1.0/docs/en/getting_started.md) documentation for more details on configurating local environment.

- [Installation](#installation)
  - [Download REDFormer Model](#download-redformer-model)
  - [Create Virtual Environment](#create-virtual-environment)
    - [Install conda package](#install-conda-package)
    - [Create conda environment](#create-conda-environment)
    - [Activate environment and install `mmdetection3d` library](#activate-environment-and-install-mmdetection3d-library)
  - [Build the Project](#build-the-project)

______________________________________________________________________

## Download REDFormer Model

Download our project repository and move into the project root directory.

```bash
cd && git clone --depth=1 -b master https://https://github.com/PurdueDigitalTwin/REDFormer
cd REDFormer
```

______________________________________________________________________

## Create Virtual Environment

Next we setup a virtual environment in `conda` for running the model.

### Install conda package

Our model runs on a local virtual environment. We recommend [Anaconda](https://docs.anaconda.com/anaconda/install/index.html) or [Miniconda](https://docs.conda.io/en/main/miniconda.html) for this purpose. Please follow their instructions to install locally.We highly recommend.

### Create conda environment

Now we create a virtual environment using conda for our model to run inside.

```bash
conda env create -f environment.yml
```

### Activate environment and install `mmdetection3d` library

Use the following steps to install the `mmdetection3d` library locally.

```bash
cd && git clone https://github.com/open-mmlab/mmdetection3d.git && cd mmdetection3d
conda activate redformer
pip install -e .
```

To quite the virtual environment, use the following scripit:

```bash
conda deactivate redformer
```

______________________________________________________________________

## Build the Project

Finally, we recommend you to install our project as a local PyPI development package. Inside the project root directory, run the following script to install the project:

```bash
pip install -e .
```

______________________________________________________________________

That's it! Now you are all set to go.
