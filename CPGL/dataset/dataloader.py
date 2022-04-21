#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  3 19:09:27 2020

@author: lin
"""

import os.path as osp
from PIL import Image
from torch.utils.data import Dataset
from torchvision import transforms
import torch
import torchvision.transforms.functional as F




class ImageLabelLoader(Dataset):

    def __init__(self, setname,data_root,csv_root, mode='train'):
        ROOT_PATH = data_root
        CSV_PATH = csv_root
        csv_path = osp.join(CSV_PATH, setname + '.csv')
        lines = [x.strip() for x in open(csv_path, 'r').readlines()][1:]

        data = []
        label = []
        lb = -1

        self.wnids = []

        for l in lines:
            name, wnid = l.split(',')
            # path = osp.join(ROOT_PATH, 'images', name)
            # path = osp.join(ROOT_PATH, 'uctrain', name)
            path = osp.join(ROOT_PATH,wnid, name)
            if wnid not in self.wnids:
                self.wnids.append(wnid)
                lb += 1
            data.append(path)
            label.append(lb)

        self.data = data
        self.label = label
        
        self.mode = mode
        self.train_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.RandomHorizontalFlip(p=0.5),
            transforms.RandomVerticalFlip(p=0.5),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        
        self.test_transform = transforms.Compose([
            transforms.Resize((224,224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                  std=[0.229, 0.224, 0.225])
        ])
        
       
    def __len__(self):
        return len(self.data)

    def __getitem__(self, i):
        path, label = self.data[i], self.label[i]
        if self.mode == 'train':
            image = self.train_transform(Image.open(path).convert('RGB'))
        elif self.mode == 'test':
            image = self.test_transform(Image.open(path).convert('RGB'))
        
        return image,label



