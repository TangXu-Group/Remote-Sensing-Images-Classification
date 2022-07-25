# EMTCAL: Efficient Multi-Scale Transformer and Cross-Level Attention Learning for Remote Sensing Scene Classification

The official code for paper "EMTCAL: Efficient Multi-Scale Transformer and Cross-Level Attention Learning for Remote Sensing Scene Classification"

<img src="https://github.com/TangXu-Group/Remote-Sensing-Images-Classification/blob/main/EMTCAL/image/framework.jpg" width="800px">



## Install dependencies
    timm, pip install timm=0.4.12
    pytorch=1.8.1
## Data
We conduct the experiments on four data sets, including UC Merced, AID, and NWPU-RESISC45. To train and test our model, you should 
    download the data set and modify image's path according to your needs.
## Training
All the configurations are in `main.py`, and you can modify them by your needs.

#### train the model
    ./main.sh


