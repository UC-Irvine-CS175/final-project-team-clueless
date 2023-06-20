# CS 175: Team Clueless Final Project Repository
[![PyTorch](https://img.shields.io/badge/PyTorch-%23EE4C2C.svg?style=flat&logo=PyTorch&logoColor=white)](https://pytorch.org/)
[![PyTorch Lightning](https://img.shields.io/badge/PyTorch%20Lightning-%23FFA500.svg?style=flat&logo=PyTorch&logoColor=white)](https://www.pytorchlightning.ai/)
[![Python](https://img.shields.io/badge/Python-%233776AB.svg?style=flat&logo=Python&logoColor=white)](https://www.python.org/)
[![Weights and Biases](https://img.shields.io/badge/Weights%20and%20Biases-%23FFBE00.svg?style=flat&logo=Weights%20and%20Biases&logoColor=white)](https://wandb.ai/)
## Contributors
- @sammypham
- @lauravalko
- @astraljdc
- @Starfractor
## Introduction
This is the official repository for the CS 175 final project from Team Clueless. Our team goal is to implement a self-supervised model that can accurately assign a pseudo-label onto an unlabeled image. By using our model and providing an image as the input, researchers can visualize similarities to existing images and labels in their dataset. Applications for our model include labelling new data that is being collected and understanding the comparable damage to already-known radiation levels and types, along with utilizing already-collected data from older datasets that would take too long to manually label.
## Enviornment Setup
The dependencies of this program relies on using the Conda environment management system. To install the dependencies for this project, run one of the following yml files, `setup\environments\cs175_cpu.yml` or `setup\environments\cs175_gpu.yml`. Make sure you are on the newly created Conda enviornment before proceeding. For additional assistance on setting up, refer to `setup\environments\README.md` for more detailed instructions. 
## Downloading Data
The data we are working with comes from the NASA BPS Microscopy Dataset, which contains images of individual nuclei of fibroblast cells from mice irradiated with either Fe particles or X-rays. The data collected comes from the AWS bucket located here: https://aws.amazon.com/marketplace/pp/prodview-6eq625wnwk4b6#usage. To download the data from this repository, you will want to run the `src\data_utils.py` and a newly created folder called `data` will be created, which contains all the data from the AWS bucket.
## Running Baseline and Deep Learning Models
This repository contains two baseline models, T-SNE and ResNet-101, and a deep learning model, SimCLR. In order to run these models on the dataset, first make sure that the NASA BPS Microscopy Dataset is downloaded locally on your machine. You can then call the specified `.py` file of the model, which are `src\model\baseline\unsupervised\pca_tsne.py`, `src\model\baseline\unsupervised\resnet101.py`, and `src\model\simclr\simclr.py`. Each model will then produce their own set of `.png` images inside the directory of their `.py` file.
