# Multi-pretext-task Prototypes Guided Dynamic Contrastive Learning Network for Few-shot Remote Sensing Scene Classification

The official code for paper "Multi-pretext-task Prototypes Guided Dynamic Contrastive Learning Network for Few-shot Remote Sensing Scene Classification"

<img src="https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/blob/main/CPGL/image/CPGL.png" width="1000px">



## Install dependencies
    pytorch>=1.7.1
## Data
We conduct the experiments on four data sets, including UC Merced, AID, and NWPU-RESISC45. To train, validate and test our model, you should 
    download the data set and modify image's path according to your needs.
## Training
All the configurations are in `train.py`, and you can modify them by your needs.

#### train and validate the model
python train.py

#### test the model
python test.py




