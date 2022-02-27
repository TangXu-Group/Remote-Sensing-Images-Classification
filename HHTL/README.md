# Deep-Hash-learning-for-Remote-Sensing-Image-Retrieval
The official codes for paper "Deep hash learning for remote sensing image retrieval"

## Install dependencies
    numpy
    opencv-python
    torch
    torchvision
## Data
We conduct the experiments on three data sets, including [UC Merced](http://weegee.vision.ucmerced.edu/datasets/landuse.html), [AID](https://captain-whu.github.io/AID/), and NWPU-RESISC45. To train and test our model, you should download the data set and modify each image's path in the `dataset/AID/.txt` or `dataset/NWPU/.txt` or `dataset/UC_Merced/.txt` (depending which data set you select to conduct the experiment)

## Training
All the configurations are in `trainerAndHash.py`, and you can modify them by your needs.

#### train the model
    python trainerAndHash.py --phase=0
