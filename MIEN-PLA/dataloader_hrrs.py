from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as transforms
import random
import torchvision
import numpy as np
from PIL import Image
from numpy.testing import assert_array_almost_equal
import os
from label_to_noise import noisify_dataset

IMG_EXTENSIONS = ['.jpg', '.jpeg', '.png', '.ppm', '.bmp', '.pgm', '.tif']

def pil_loader(path):    # 一般采用pil_loader函数。
    with open(path, 'rb') as f:
        with Image.open(f) as img:
            return img.convert('RGB')

def default_loader(path):
  
    return pil_loader(path)

def find_classes(dir):
    classes = [d.name for d in os.scandir(dir) if d.is_dir()]
    classes.sort()
    class_to_idx = {cls_name: i for i, cls_name in enumerate(classes)}
    print
    return classes, class_to_idx


def make_dataset(dir, classes, class_to_idx, extensions,is_valid_file=None):
    instances = []
    directory = os.path.expanduser(dir)
    both_none = extensions is None and is_valid_file is None
    both_something = extensions is not None and is_valid_file is not None
    if both_none or both_something:
        raise ValueError("Both extensions and is_valid_file cannot be None or not None at the same time")
    if extensions is not None:
        def is_valid_file(x):
            return has_file_allowed_extension(x, extensions)
    for target_class in sorted(class_to_idx.keys()):
        class_index = class_to_idx[target_class]
        target_dir = os.path.join(directory, target_class)
        if not os.path.isdir(target_dir):
            continue
        for root, _, fnames in sorted(os.walk(target_dir, followlinks=True)):
            for fname in sorted(fnames):
                path = os.path.join(root, fname)
                if is_valid_file(path):
                    item = path, class_index
                    instances.append(item)
    return instances

def has_file_allowed_extension(filename, extensions):
    """Checks if a file is an allowed extension.

    Args:
        filename (string): path to a file

    Returns:
        bool: True if the filename ends with a known image extension
    """
    # 将文件的名变成小写
    filename_lower = filename.lower()

    # endswith() 方法用于判断字符串是否以指定后缀结尾
    # 如果以指定后缀结尾返回True，否则返回False
    return any(filename_lower.endswith(ext) for ext in extensions)

class NewImageFolder(Dataset):
    """默认图像数据目录结构
    root
    .
    ├──dog
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──cat  
    |   ├──001.png
    |   ├──002.png
    |   └──...
    └──...
    """
    def __init__(self, root, train=True, transform=None,
                 loader=default_loader, noise_type='clean', noise_ratio=0.1, nb_classes=45):
        super(NewImageFolder,self).__init__()
        classes, class_to_idx = find_classes(root)#寻找类和对应的类别号
        imgs = make_dataset(root, classes, class_to_idx, extensions=IMG_EXTENSIONS)#图片地址和类别
        if len(imgs) == 0:
            raise(RuntimeError("Found 0 images in subfolders of: " + root + "\n"
                               "Supported image extensions are: " + ",".join(IMG_EXTENSIONS)))
        if not train:
            assert noise_type == 'clean', f'In test mode, noise_type should be clean, but got {noise_type}!'

        self.root = root
        self.imgs = imgs
        self.classes = classes
        self.class_to_idx = class_to_idx
        self.transform = transform
        self.loader = loader
        self.train = train  
        #### 加噪声
        self.nb_classes = nb_classes
        self.noise_type = noise_type
        self.noise_ratio = noise_ratio
        self.targets = list(np.asarray([img[1] for img in self.imgs]))
        if self.train and (noise_type != 'clean'):
            self.train_labels = np.asarray([img[1] for img in self.imgs])
            self.noisy_labels, self.actual_noise_rate = noisify_dataset(self.nb_classes, self.train_labels, self.noise_type, self.noise_ratio)

    def __len__(self):
        return len(self.imgs)

    def __getitem__(self, index):
        """
        index (int): Index
        Returns:tuple: (image, target) where target is class_index of the target class.
        """

        if self.train and self.noise_type != 'clean':
            path, clean_target = self.imgs[index]
            img = self.loader(path)
            target = self.noisy_labels[index]
            img1 = self.transform[0](img)
            img2 = self.transform[1](img)
            img3 = self.transform[2](img)
            return {'index': index, 'data1': img1,'data2': img2,'data3': img3, 'noise_label': target,'true_label':clean_target}
        else:
            path, clean_target = self.imgs[index]
            img = self.loader(path)
            target = clean_target
            img = self.transform(img)
            return {'index': index, 'data': img,'true_label':clean_target}
