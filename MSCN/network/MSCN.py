import torch
import torch.nn as nn
import torch.nn.functional as F
from einops import rearrange
from math import sqrt
import cv2
from torchvision.models import resnet34,resnet50
# from network.attention import Attention0
from network.sccov import GroupCBAMEnhancer as GCE

class MSC(nn.Module):
    def __init__(self,dim,num_heads=8,topk=True,kernel = [3,5,7],s = [1,1,1],pad = [1,2,3],
                 qkv_bias=False,qk_scale=None,attn_drop_ratio=0.,proj_drop_ratio=0.,k1 = 2, k2 =3):
        super(MSC, self).__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q    = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv   = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop_ratio)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop_ratio)
        self.k1   = k1
        self.k2   = k2
        
        self.attn1 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        self.attn2 = torch.nn.Parameter(torch.tensor([0.5]), requires_grad=True)
        # self.attn3 = torch.nn.Parameter(torch.tensor([0.3]), requires_grad=True)

        self.avgpool1 = nn.AvgPool2d(kernel_size=kernel[0],stride=s[0],padding=pad[0])
        self.avgpool2 = nn.AvgPool2d(kernel_size=kernel[1],stride=s[1],padding=pad[1])
        self.avgpool3 = nn.AvgPool2d(kernel_size=kernel[2],stride=s[2],padding=pad[2])

        self.layer_norm = nn.LayerNorm(dim)
        
        self.topk = topk # False True

    def forward(self, x,y):
        # x0 = x
        y1 = self.avgpool1(y)
        y2 = self.avgpool2(y)
        y3 = self.avgpool3(y)
        # y = torch.cat([y1.flatten(-2,-1),y2.flatten(-2,-1),y3.flatten(-2,-1)],dim = -1)
        y = y1+y2+y3
        y = y.flatten(-2,-1)
    
        y = y.transpose(1, 2)
        y = self.layer_norm(y)
        x = rearrange(x,'b c h w -> b (h w) c')
        # y = rearrange(y,'b c h w -> b (h w) c')
        B, N1, C = y.shape
        # print(y.shape)
        kv  = self.kv(y).reshape(B, N1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
        # qkv = self.qkv(x).reshape(B, N, 3, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        B, N, C = x.shape
        q   = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0,2,1,3)
        
        attn = (q @ k.transpose(-2, -1)) * self.scale
    
        # print(self.k1,self.k2)    
        mask1 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1/self.k1), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask1.scatter_(-1, index, 1.)
        attn1 = torch.where(mask1 > 0, attn, torch.full_like(attn, float('-inf')))
        attn1 = attn1.softmax(dim=-1)
        attn1 = self.attn_drop(attn1)
        out1 = (attn1 @ v)

        mask2 = torch.zeros(B, self.num_heads, N, N1, device=x.device, requires_grad=False)
        index = torch.topk(attn, k=int(N1/self.k2), dim=-1, largest=True)[1]
        # print(index[0,:,48])
        mask2.scatter_(-1, index, 1.)
        attn2 = torch.where(mask2 > 0, attn, torch.full_like(attn, float('-inf')))
        attn2 = attn2.softmax(dim=-1)
        attn2 = self.attn_drop(attn2)
        out2 = (attn2 @ v)

        out = out1 * self.attn1 + out2 * self.attn2 #+ out3 * self.attn3
        # out = out1 * self.attn1 + out2 * self.attn2

        x = out.transpose(1, 2).reshape(B, N, C)
   
        x = self.proj(x)
        x = self.proj_drop(x)
        hw = int(sqrt(N))
        x = rearrange(x,'b (h w) c -> b c h w',h=hw,w=hw)
        # x = x + x0
        return x

class MSCN(nn.Module):
    def __init__(self,num_classes = 45,res = 50,k1 = 2,k2 = 3,g = 8):
        super(MSCN, self).__init__()
        # print(f'att = {att}')
        if res == 34:
            self.resnet = resnet34()
            dim = [64,128,256,512]
            self.dim = dim
        elif res == 50:
            self.resnet = resnet50()
            dim = [256,512,1024,2048]
            self.dim = dim
        dim_b = self.dim[1]
        dim_fc = self.dim[-1]

        self.msc1 = MSC(dim=dim_b,kernel=[3,5,7],pad=[1,2,3],k1=k1,k2=k2)
        self.msc2 = MSC(dim=dim_b,kernel=[3,5,7],pad=[1,2,3],k1=k1,k2=k2)
        self.msc3 = MSC(dim=dim_b,kernel=[3,5,7],pad=[1,2,3],k1=k1,k2=k2)

        self.gce = GCE(dim_fc,g,0,0)
        
        self.cov1 = nn.Conv2d(dim[0],dim_b,3,2,1)
        self.cov2 = nn.Conv2d(dim[1],dim_b,3,2,1)
        self.cov3 = nn.Conv2d(dim[2],dim_b,3,2,1)
        self.cov4 = nn.Conv2d(dim[3],dim_b,1)

        self.avgpool = nn.AdaptiveAvgPool2d(1)
        self.fc = nn.Linear(dim_fc,num_classes)

    def reset_head(self,num_classes):
        self.fc = nn.Linear(self.dim[-1],num_classes)

    def forward(self,x):
        x = self.resnet.conv1(x)
        x = self.resnet.bn1(x)
        x = self.resnet.relu(x)
        x = self.resnet.maxpool(x)
        x = self.resnet.layer1(x)
        x1 = self.cov1(x)
        x = self.resnet.layer2(x)
        x2 = self.cov2(x)
        x = self.resnet.layer3(x)
        x3 = self.cov3(x)
        x = self.resnet.layer4(x)
        x4 = self.cov4(x)

        x1 = self.msc1(x4,x1)
        x2 = self.msc2(x4,x2)
        x3 = self.msc3(x4,x3)
        
        x  = torch.cat([x4,x1,x2,x3],dim=1)
        x  = self.gce(x)
        
        x  = self.avgpool(x)
        
        x = torch.flatten(x, 1)
        x = self.fc(x)
        
        return x
    
if __name__ == '__main__':
    print('net test')   
    a = torch.randn(1, 3, 224, 224)
    model = MSCN(res=50)
    # model = resnet50()
    # print(model.resnet.fc)
    # from fvcore.nn import FlopCountAnalysis
    # flops = FlopCountAnalysis(model, a)
    # print("FLOPs: %.2fG" % (flops.total()/1e9))
    total = sum([param.nelement() for param in model.parameters()])
    print("Number of parameter: %.2fM" % (total/1e6))
    b = model(a)
    print(b.shape)