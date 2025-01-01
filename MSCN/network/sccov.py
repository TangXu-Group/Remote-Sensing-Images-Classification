import torch
import torch.nn.functional as F
import torch.nn as nn 

class ChannelAttention(nn.Module):
    def __init__(self, in_channels, reduction_ratio=16):
        super(ChannelAttention, self).__init__()
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Sequential(
            nn.Linear(in_channels, in_channels // reduction_ratio),
            nn.ReLU(),
            nn.Linear(in_channels // reduction_ratio, in_channels),
            nn.Sigmoid()
        )
    def forward(self, x):
        avg_out = self.avg_pool(x).view(x.size(0), -1)
        channel_attention = self.fc(avg_out).view(x.size(0), x.size(1), 1, 1)
        return x * channel_attention

class SpatialAttention(nn.Module):
    def __init__(self, in_channels):
        super(SpatialAttention, self).__init__()
        self.conv = nn.Conv2d(in_channels, 1, kernel_size=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        attention = self.conv(x)
        attention = self.sigmoid(attention)
        x = x * attention
        return x

class CBAM(nn.Module):
    def __init__(self, in_channels, reduction_ratio=4):
        super(CBAM, self).__init__()
        self.channel_attention = ChannelAttention(in_channels, reduction_ratio)
        self.spatial_attention = SpatialAttention(in_channels)

    def forward(self, x):
        x = self.channel_attention(x)
        x = self.spatial_attention(x)
        return x
        

class GroupCBAMEnhancer(nn.Module):
    def __init__(self,channel,group = 8,cov1=1,cov2=1):
        super().__init__()
        self.cov1 = None
        self.cov2 = None
        if cov1 != 0:
            self.cov1 = nn.Conv2d(channel,channel,kernel_size=1)    
        self.group = group
        cbam = []
        for i in range(self.group):
            cbam_ = CBAM(channel//group)
            cbam.append(cbam_)
        
        self.cbam = nn.ModuleList(cbam)
        self.sigomid = nn.Sigmoid()
        if cov2 != 0:
            self.cov2 = nn.Conv2d(channel,channel,kernel_size=1)

    def forward(self,x):
        x0 = x
        if self.cov1 != None:
            x = self.cov1(x)
        y = torch.split(x,x.size(1)//self.group,dim=1)
        mask = []
        for y_,cbam in zip(y,self.cbam):
            y_ = cbam(y_)
            y_ = self.sigomid(y_)

            mk = y_

            mean = torch.mean(y_,[1,2,3])
            mean = mean.view(-1,1,1,1)

            # mean = torch.mean(y_,[2,3])
            # mean = mean.view(mean.size(0),mean.size(1),1,1)
            
            gate = torch.ones_like(y_) * mean
            mk = torch.where(y_ > gate, 1, y_)

            mask.append(mk)
        mask = torch.cat(mask,dim=1)
        # print(mask.shape)
        x = x * mask
        if self.cov2 != None:
            x = self.cov2(x)
        x = x + x0
        return x

if __name__ == '__main__':
    x       = torch.randn(1,4,16,16)
    model   = GroupCBAMEnhancer(4,1)
    print(model(x).shape)