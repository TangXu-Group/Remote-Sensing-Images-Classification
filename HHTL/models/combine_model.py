import torch.nn as nn
import numpy as np
from models.slic_model import slic_VisionTransformerc
from models.model import VisionTransformerc


class CombineVisionTransformerc(nn.Module):
    def __init__(self, config, img_size=384, num_classes=21, zero_head=False, vis=False, pretrained_dir=None):
        super(CombineVisionTransformerc, self).__init__()
#         self.num_classes = num_classes
#         self.zero_head = zero_head
#         self.pretrained_dir = pretrained_dir

        self.yuan_transformer = VisionTransformerc(config, img_size, zero_head = True, num_classes = num_classes)
        self.slic_transformer = slic_VisionTransformerc(config, img_size, zero_head = True, num_classes = num_classes)

        self.yuan_transformer.load_from(np.load(pretrained_dir))
        self.slic_transformer.load_from(np.load(pretrained_dir))

    def forward(self, x , y):
        out_yuan,ca,feature_yuan = self.yuan_transformer(x)
        out_slic,ac,feature_slic = self.slic_transformer(y)

        out = out_yuan + out_slic
        out = out / 2.0

        return out, feature_yuan, feature_slic








