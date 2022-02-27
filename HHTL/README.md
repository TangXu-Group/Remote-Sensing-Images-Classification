# Homo-Heterogenous Transformer Learning Framework for Remote Sensing Scene Classification
The official codes for paper "Homo-Heterogenous Transformer Learning Framework for Remote Sensing Scene Classification"

## Install dependencies
    timm
    pytorch>=1.7
## Data
We conduct the experiments on three data sets, including [UC Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html), [AID](https://captain-whu.github.io/AID/), and NWPU-RESISC45. To train and test our model, you should download the data set and modify each image's path in the `dataset/AID/.txt` or `dataset/NWPU/.txt` or `dataset/UC_Merced/.txt` (depending which data set you select to conduct the experiment)

## Training

