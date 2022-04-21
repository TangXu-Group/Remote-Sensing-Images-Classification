#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Sun Mar 14 15:47:57 2021

@author: lin
"""

import torch
import torch.nn as nn
import torch.nn.functional as F
from torchvision import models


class softmax(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(softmax,self).__init__()
        
        self.fc = nn.Linear(input_dim,num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.fc(x)
        return out

class s_auxiliary_softmax(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(s_auxiliary_softmax,self).__init__()
        
        self.fc = nn.Linear(input_dim,num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.fc(x)
        return out

class v_auxiliary_softmax(nn.Module):
    def __init__(self,input_dim,num_classes):
        super(v_auxiliary_softmax,self).__init__()
        
        self.fc = nn.Linear(input_dim,num_classes)
        
        self.relu = nn.ReLU(inplace=True)
    def forward(self,x):
        out = self.fc(x)
        return out

class ConvLayer(nn.Module):
    def __init__(self,input_dim,output_dim):
        super(ConvLayer,self).__init__()
        self.convfc = nn.Conv2d(input_dim,output_dim,kernel_size=1)
        self.bn = nn.BatchNorm2d(output_dim)
        self.relu = nn.ReLU(inplace=True)
        
    def forward(self,x):
        out = self.convfc(x)
        out = self.bn(out)
        out = self.relu(out)
        return out



    
class FEM(nn.Module):
    def __init__(self,kernel_num):
        super(FEM,self).__init__()
        # self.agpool3_3 = nn.AvgPool2d(kernel_size=4,stride=4)
        # self.agpool4_3 = nn.AvgPool2d(kernel_size=2,stride=2)
        self.N = 7
        self.kernel_num = kernel_num
        # self.agpool3_3 = nn.AdaptiveAvgPool2d((self.N,self.N))
        # self.agpool4_3 = nn.AdaptiveAvgPool2d((self.N,self.N))
        self.conv = nn.Conv2d(896,self.kernel_num,kernel_size=1)
        self.gapool = nn.AdaptiveAvgPool2d((1,1))
        self.relu = nn.ReLU(inplace=True)
    def forward(self,conv3_3,conv4_3,conv5_3,N):
        agpool3_3 = nn.AdaptiveAvgPool2d((N,N))
        agpool4_3 = nn.AdaptiveAvgPool2d((N,N))
        conv3_3 = F.normalize(agpool3_3(conv3_3))#128xNxN,N=smallest_scale
        conv4_3 = F.normalize(agpool4_3(conv4_3))#256xNxN
        conv5_3 = F.normalize(conv5_3)#512xNxN
        fusion_feature = torch.cat((torch.cat((conv3_3,conv4_3),1),conv5_3),1)#1280xNxN
        fusion_feature = self.conv(fusion_feature)#kernel_numxNxN
        fusion_feature = self.relu(fusion_feature)
        fusion_feature = self.gapool(fusion_feature)#kernel_numx1x1
        fusion_feature = torch.flatten(fusion_feature,1)#kernel_num
        # final_representation = torch.cat((fusion_feature,fc2_out),1)
        return fusion_feature
    
    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.xavier_uniform(m.weight)

class BasicBlock(nn.Module):#18-layer和34-layer
    #在18和34layer每个conv中第一层和第二层的卷积核个数相同
    expansion = 1

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        #in_channel输入特征矩阵的深度，out_channel输出特征矩阵的深度，downsample对于虚线残差结构
        super(BasicBlock, self).__init__()
        #这里的conv1和conv2是convi_x中的两个卷积层
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=3, stride=stride, padding=1, bias=False)
        # stride=1：output = (input-3+2*1)/1+1=input
        # stride=2：output = (input-3+2*1)/2+1 = input/2+0.5 = input/2(向下取整)
        # 这里的output指的是feature map的宽和高
        self.bn1 = nn.BatchNorm2d(out_channel)
        self.relu = nn.ReLU()
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=1, padding=1, bias=False)
        self.bn2 = nn.BatchNorm2d(out_channel)
        self.downsample = downsample

    def forward(self, x):
        #捷径分支上的输出值
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):#50-layer、101-layer和152-layer
    #对于50、101、152layer，每个conv的第三层卷积核个数是第二层的四倍
    expansion = 4

    def __init__(self, in_channel, out_channel, stride=1, downsample=None):
        super(Bottleneck, self).__init__()
        self.conv1 = nn.Conv2d(in_channels=in_channel, out_channels=out_channel,
                                kernel_size=1, stride=1, bias=False)  # squeeze channels
        self.bn1 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv2 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel,
                                kernel_size=3, stride=stride, bias=False, padding=1)
        self.bn2 = nn.BatchNorm2d(out_channel)
        # -----------------------------------------
        self.conv3 = nn.Conv2d(in_channels=out_channel, out_channels=out_channel*self.expansion,
                                kernel_size=1, stride=1, bias=False)  # unsqueeze channels
        self.bn3 = nn.BatchNorm2d(out_channel*self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample

    def forward(self, x):
        identity = x
        if self.downsample is not None:
            identity = self.downsample(x)

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        out += identity
        out = self.relu(out)

        return out


class ResNet(nn.Module):

    def __init__(self, block, blocks_num, num_classes=1000, include_top=True):
        super(ResNet, self).__init__()
        self.include_top = include_top
        self.in_channel = 64#通过maxpooling后输入特征矩阵的深度

        self.conv1 = nn.Conv2d(3, self.in_channel, kernel_size=7, stride=2,
                                padding=3, bias=False)#3表示R、G、B三个通道
        #output = (input-7+2*3)/2 = input/2 + 0.5(向下取整) = input/2
        self.bn1 = nn.BatchNorm2d(self.in_channel)
        self.relu = nn.ReLU(inplace=True)
        self.maxpool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        #18、34layer的conv2_x的输入feature map不需要改变深度，而50、101、152layer则需要改变深度
        #使用默认stride=1是因为经过maxpooling后feature map的高宽为56x56，不需要改变shape
        self.layer1 = self._make_layer(block, 64, blocks_num[0])
        self.layer2 = self._make_layer(block, 128, blocks_num[1], stride=2)
        self.layer3 = self._make_layer(block, 256, blocks_num[2], stride=2)
        self.layer4 = self._make_layer(block, 512, blocks_num[3], stride=2)
        self.convlayer = ConvLayer(512,512)
        if self.include_top:
            self.avgpool = nn.AdaptiveAvgPool2d((1, 1))  # 这里说明ResNet可以接受任意尺寸图像的输入，output size = (1, 1),相当于全局平均池化
            # self.avgpool = nn.AdaptiveMaxPool2d((1, 1))
            self.fc = nn.Linear(512 * block.expansion, num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

    def _make_layer(self, block, channel, block_num, stride=1):
        downsample = None
        if stride != 1 or self.in_channel != channel * block.expansion:
            downsample = nn.Sequential(
                nn.Conv2d(self.in_channel, channel * block.expansion, kernel_size=1, stride=stride, bias=False),
                nn.BatchNorm2d(channel * block.expansion))

        layers = []
        layers.append(block(self.in_channel, channel, downsample=downsample, stride=stride))
        self.in_channel = channel * block.expansion#注意这里的倍数关系

        for _ in range(1, block_num):#从1开始是因为有虚线shortcut的前面已添加
            layers.append(block(self.in_channel, channel))

        return nn.Sequential(*layers)#通过*将layer转化为可变参数

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = self.relu(x)
        x = self.maxpool(x)

        x = self.layer1(x)
        x = self.layer2(x)
        x = self.layer3(x)
        x = self.layer4(x)

        if self.include_top:
            x = self.avgpool(x)
            # x = torch.flatten(x, 1)
            x = self.convlayer(x)
            x = torch.flatten(x,1)

        return x

def resnet18(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [2, 2, 2, 2], num_classes=num_classes, include_top=include_top)

def resnet34(num_classes=1000, include_top=True):
    return ResNet(BasicBlock, [3, 4, 6, 3], num_classes=num_classes, include_top=include_top)


def resnet101(num_classes=1000, include_top=True):
    return ResNet(Bottleneck, [3, 4, 23, 3], num_classes=num_classes, include_top=include_top)

# if __name__=='__main__':
#     data = torch.randn([2,3,224,224]).cuda()
#     net = ResNet18().cuda()
#     fem = FEM(kernel_num=2048).cuda()
#     c3,c4,c5,fc_out = net(data)
#     print(c3.size())
#     print(c4.size())
#     print(c5.size())
#     print(fc_out.size())
#     fusion_feature = fem(c3,c4,c5)
#     print(fusion_feature.size())
    
    
    