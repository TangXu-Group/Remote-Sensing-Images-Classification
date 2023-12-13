#用来做h和transformer block数量的参数分析
from re import S
import torch
import torch.nn as nn
from torchvision.models import resnet18
from timm.models.layers import DropPath
import numpy as np


class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self,
                 in_features,
                 hidden_features=None,
                 out_features=None,
                 act_layer=nn.GELU,
                 drop=0.):
        super().__init__()
        out_features = out_features or in_features
        hidden_features = hidden_features or in_features
        self.fc1 = nn.Linear(in_features, hidden_features)
        self.act = act_layer()
        self.fc2 = nn.Linear(hidden_features, out_features)
        self.drop = nn.Dropout(drop)

    def forward(self, x):
        x = self.fc1(x)
        x = self.act(x)
        x = self.drop(x)
        x = self.fc2(x)
        x = self.drop(x)
        return x


class DimReduceLayer(nn.Module):

    def __init__(self, in_channels, out_channels):
        super(DimReduceLayer, self).__init__()
        layers = []
        layers.append(
            nn.Conv2d(in_channels,
                      out_channels,
                      1,
                      stride=1,
                      padding=0,
                      bias=False))
        # layers.append(nn.BatchNorm2d(out_channels))

        # layers.append(nn.ReLU(inplace=True))

        self.layers = nn.Sequential(*layers)

    def forward(self, x):
        return self.layers(x)


class Attention(nn.Module):
    "Implementation of self-attention"

    def __init__(self,
                 dim=512,
                 num_heads=4,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 proj_drop=0.,
                 sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim**-0.5

        self.qkv = nn.Linear(dim, dim * 3, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

        self.norm = nn.LayerNorm(dim)
        self.convMain = nn.ModuleList(
            [DimReduceLayer(1, 1) for _ in range(self.num_heads)])
        self.convRest = nn.ModuleList([
            DimReduceLayer(self.num_heads - 1, 1)
            for _ in range(self.num_heads)
        ])

    def CIM(self, attn_h):  #加不加激活
        attn_h_new = []
        for index in range(self.num_heads):
            main = attn_h[index]
            res = torch.cat(
                [attn_h[j] for j in range(len(attn_h)) if j != index], 1)
            qk_main = self.convMain[index](main)
            qk_res = self.convRest[index](res)
            qk_interaction = qk_main + qk_res
            attn_h_new.append(qk_interaction)
        attn_h_return = torch.cat(attn_h_new, 1)

        return attn_h_return

    def forward(self, x):
        B, N, C = x.shape

        qkv = self.qkv(x).reshape(B, N, 3, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)

        q, k, v = qkv[0], qkv[1], qkv[2]
        attn = (q @ k.transpose(-2, -1)) * self.scale  #B H T T
        '''融合'''
        attn_h = [attn[:, i, :, :].unsqueeze(1) for i in range(self.num_heads)]
        attn = self.CIM(attn_h)
        '''融合结束'''
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)
        return x


class TransformerBlock(nn.Module):
    """
    Implementation of Transformer,
    """

    def __init__(self,
                 dim,
                 num_heads,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 drop=0.,
                 attn_drop=0.,
                 drop_path=0.,
                 act_layer=nn.GELU,
                 norm_layer=nn.LayerNorm,
                 sr_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim,
                              num_heads=num_heads,
                              qkv_bias=qkv_bias,
                              qk_scale=qk_scale,
                              attn_drop=attn_drop,
                              proj_drop=drop,
                              sr_ratio=sr_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(
            drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim,
                       hidden_features=mlp_hidden_dim,
                       act_layer=act_layer,
                       drop=drop)

    def forward(self, x):  #b,128,784
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x


def l2_normalization(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm + 1e-8)
    return x


class MIEN(nn.Module):

    def __init__(self,
                 num_class=45,
                 h=4,
                 dim=512,
                 num_heads=12,
                 mlp_ratio=4.,
                 qkv_bias=False,
                 qk_scale=None,
                 attn_drop=0.,
                 drop_path_rate=0.1,
                 drop_rate=0.,
                 numblocks=1):
        super(MIEN, self).__init__()
        self.backbone = resnet18(pretrained=True)
        self.transformer = nn.ModuleList([])
        for _ in range(numblocks):
            self.transformer.append(
                TransformerBlock(dim=512,
                                 num_heads=h,
                                 mlp_ratio=mlp_ratio,
                                 qkv_bias=qkv_bias,
                                 qk_scale=qk_scale,
                                 attn_drop=attn_drop,
                                 drop_path=drop_path_rate,
                                 ))
        self.pos_drops = nn.Dropout(p=drop_rate)
        self.relu1 = nn.ReLU()
        self.avg1 = nn.AvgPool2d(4, 4)
        self.avg2 = nn.AvgPool2d(2, 2)
        self.fc = nn.Linear(512 + 512, num_class)
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1))
        self.avgpool7 = nn.AdaptiveAvgPool2d((7, 7))
        self.conv11 = nn.Conv2d(64, 512, kernel_size=1, stride=1)
        self.conv22 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
        self.conv33 = nn.Conv2d(256, 512, kernel_size=1, stride=1)

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)  #(b,64,56,56)
        x = self.backbone.layer1(x)  #(b,64,56,56)
        f1 = x  #[1, 64, 56, 56]
        x = self.backbone.layer2(x)  #(b,128,28,28)
        f2 = x  #[[1, 128, 28, 28]]

        x = self.backbone.layer3(x)  #(b,256,14,14)
        f3 = x  #[1, 256, 14, 14]

        x = self.backbone.layer4(x)  #(b,512,7,7)
        f4 = x  #[1, 512, 7, 7]

        f1 = self.conv11(f1)
        f1 = self.avgpool7(f1)
        f2 = self.conv22(f2)
        f2 = self.avgpool7(f2)
        f3 = self.conv33(f3)
        f3 = self.avgpool7(f3)
        f1 = l2_normalization(f1)
        f2 = l2_normalization(f2)
        f3 = l2_normalization(f3)
        f4 = l2_normalization(f4)
        #         f = torch.cat((f1,f2,f3,f4),dim=3).reshape(B,512,14,14)
        #         ff1 = torch.cat((f1,f2),dim=2)
        #         ff2 = torch.cat((f3,f4),dim=2)
        ff1 = torch.cat((f1, f2), dim=2)
        ff2 = torch.cat((f3, f4), dim=2)

        f = torch.cat((ff1, ff2), dim=3)

        f_0 = f
        f_b, f_c, f_h, f_w = f.shape
        f = f.flatten(2).transpose(1, 2)
        for tran in self.transformer:
            f = tran(f)
        f = f.transpose(1, 2)
        f = f.reshape(f_b, f_c, f_h, f_w)
        #         f = f + f_0

        #         f = l2_normalization(f)
        f = self.avgpool(f).flatten(1)
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = torch.cat((x, f), dim=1)
        x = self.fc(x)

        return x