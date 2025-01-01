This repository provides the code for the method in our paper 'Multiscale Sparse Cross-Attention Network for Remote Sensing Scene Classification'. (TGRS2024)

If you have any questions, you can send me an email. My mail address is 22171214702@stu.xidian.edu.cn.

# Datasets
We conduct experiments on the UCM, AID, and NWPU datasets. To train and test our model, you should download the required data set and modify the corresponding parameters in main.py to meet your needs.

# requirements
torch==1.7.1
torchvision==0.8.2
timm==0.4.12
fvcore
einops
submitit

# Train
By executing the following command, the experimental results can be obtained.

python -m torch.distributed.launch --nproc_per_node=1 --use_env main.py --model MSCN --batch-size 128 --data-set UCM --train_dir {path of training set} --test_dir {path of testing set} --k1 2 --k2 6 --g 8
