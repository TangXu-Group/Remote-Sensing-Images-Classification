import torch.nn as nn
import torch
from cl_data_generator import Dy_Data_Generator
from Dynamic_Parameter import DAM

class S_Classifier(nn.Module):
    def __init__(self, z_dim=1600, trans_num=4):
        super(S_Classifier,self).__init__()
        self.scale_classifier = nn.Sequential(nn.Linear(z_dim, trans_num),
                                                nn.ReLU())
    def forward(self,x):
        pred = self.scale_classifier(x)
        return pred

class R_Classifier(nn.Module):
    def __init__(self, z_dim=1600, trans_num=4):
        super(R_Classifier,self).__init__()
        self.scale_classifier = nn.Sequential(nn.Linear(z_dim, trans_num),
                                                nn.ReLU())
    def forward(self,x):
        pred = self.scale_classifier(x)
        return pred

def conv_block(in_channels, out_channels):
    bn = nn.BatchNorm2d(out_channels)
    nn.init.uniform_(bn.weight) # for pytorch 1.2 or later
    return nn.Sequential(
        nn.Conv2d(in_channels, out_channels, 3, padding=1),
        bn,
        nn.ReLU(inplace=True),
        nn.MaxPool2d(2)
    )


class Convnet(nn.Module):

    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(Convnet,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )

        # for m in self.modules():
        #     if isinstance(m, nn.Conv2d):
        #         nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
        #     elif isinstance(m, nn.BatchNorm2d):
        #         nn.init.constant_(m.weight, 1)
        #         nn.init.constant_(m.bias, 0)

    def forward(self, x):
        x = self.encoder(x)
        # return x.view(x.size(0), -1)
        x = nn.AdaptiveMaxPool2d((5,5))(x)
        return x.view(x.size(0), -1)

class Encoder(nn.Module):
    def __init__(self, x_dim=3, hid_dim=64, z_dim=64):
        super(Encoder,self).__init__()
        self.encoder = nn.Sequential(
            conv_block(x_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, hid_dim),
            conv_block(hid_dim, z_dim),
        )
        self.rot_classifier = R_Classifier()
        self.scale_classifier = S_Classifier()
        self.scale_dynamic = DAM(1600, 4)
        self.rot_dynamic = DAM(1600, 4)
    
    def forward(self, x):
        x = self.encoder(x)
        # return x.view(x.size(0), -1)
        x = nn.AdaptiveMaxPool2d((5,5))(x)
        x1 = x.view(x.size(0),-1)
        # rot_pred = self.rot_classifier(x1)
        # scale_pred = self.scale_classifier(x1)
        # scale_sample = self.scale_dynamic(x1)
        # rot_sample = self.rot_dynamic(x1)
        return x1



