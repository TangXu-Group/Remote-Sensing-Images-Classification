import os
# from cv2 import resize
import torch.nn as nn
import torch.optim as optim
import torchvision
from dataloader_hrrs import NewImageFolder
def build_transform(rescale_size=512, crop_size=448):

    train_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((rescale_size,rescale_size)),
        torchvision.transforms.RandomHorizontalFlip(),
        torchvision.transforms.RandomVerticalFlip(),
#         torchvision.transforms.RandomApply([torchvision.transforms.ColorJitter(0.4, 0.4, 0.4, 0.1)], p=0.8),
#         torchvision.transforms.RandomGrayscale(p=0.2),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    test_transform = torchvision.transforms.Compose([
        torchvision.transforms.Resize((rescale_size,rescale_size)),
        torchvision.transforms.ToTensor(),
        torchvision.transforms.Normalize(mean=(0.485, 0.456, 0.406), std=(0.229, 0.224, 0.225))
    ])
    train_transforms = [train_transform,train_transform,test_transform]
    return {'train': train_transforms, 'test': test_transform}

def build_hrrs_dataset(train_root,test_root ,train_transforms, test_transform,noise_ratio,n,noise_type):
    train_data = NewImageFolder(root=train_root,train=True, transform=train_transforms,noise_type=noise_type,noise_ratio=noise_ratio,nb_classes=n)
    test_data  = NewImageFolder(root=test_root,train=False, transform=test_transform,noise_type='clean',nb_classes=n)
    return {'train': train_data, 'test': test_data}

