import torch
from torch.utils.data.dataset import Dataset
from PIL import Image
import numpy as np


def read_txt(path):
    imgs, labels = [], []
    with open(path, 'r') as f:
        for line in f.readlines():
            im, label = line.strip().split(' ')
            imgs.append(im)
            labels.append(int(label))

    return imgs, labels

class MyCustomDataset(Dataset):
    def __init__(self, root_path, class_num=21, transform=None):
        self.txt_path = root_path
        self.img_path, self.img_label = read_txt(root_path)
        self.transform=transform

    def __getitem__(self, index):
        img_path = self.img_path[index]
        img = Image.open(img_path)
        if self.transform is not None:
            img = self.transform(img)

        label = np.array(self.img_label[index])
        label = torch.from_numpy(label).type(torch.long)

        return img, label

    def __len__(self):
        return len(self.img_path)

