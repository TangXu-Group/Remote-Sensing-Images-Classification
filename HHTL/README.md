# Homo-Heterogenous Transformer Learning Framework for Remote Sensing Scene Classification

The official code for paper "Homo-Heterogenous Transformer Learning Framework for Remote Sensing Scene Classification"

<img src="https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/blob/main/HHTL/image/framework.png" width="900px">



## Install dependencies
    timm, pip install timm=0.3.4
    pytorch>=1.7.1
## Data
We conduct the experiments on four data sets, including UC Merced, AID, NWPU-RESISC45, and RSSDIVCS. To train and test our model, you should 
    download the data set and modify image's path according to your needs.
## Training
All the configurations are in `main.py`, and you can modify them by your needs.

#### train the model
    ./main.sh

