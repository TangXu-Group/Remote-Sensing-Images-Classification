import torch
import torch.nn as nn
from torchvision.models import resnet34
from timm.models.layers import DropPath


class Mlp(nn.Module):
    "Implementation of MLP"

    def __init__(self, in_features, hidden_features=None,
                 out_features=None, act_layer=nn.GELU,
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


class Attention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim=512, num_heads=4, qkv_bias=False, qk_scale=None, attn_drop=0., proj_drop=0., sr_ratio=2):
        super().__init__()
        assert dim % num_heads == 0, f"dim {dim} should be divided by num_heads {num_heads}."
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)
        self.sr1 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=1, stride=2)
        self.sr2 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=3, stride=sr_ratio, padding=1, dilation=1)
        self.sr3 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=3, stride=sr_ratio, padding=2, dilation=2)
        self.sr4 = nn.Conv2d(int(dim/4), int(dim/4), kernel_size=3, stride=sr_ratio, padding=3, dilation=3)

        self.pos1 = PosCNN(int(dim/4), int(dim/4))
        self.pos2 = PosCNN(int(dim/4), int(dim/4))
        self.pos3 = PosCNN(int(dim/4), int(dim/4))
        self.pos4 = PosCNN(int(dim/4), int(dim/4))
        
        
        self.norm = nn.LayerNorm(dim)

    def forward(self, x):
        B, N, C = x.shape
        H = W = int(N ** (1/2))
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)
        x = x.transpose(-2,-1).reshape(B, C, H, W)
        x1 = x[:,:int(C/4),:,:]
        x2 = x[:,int(C/4):int(C/4)*2,:,:]
        x3 = x[:,int(C/4)*2:int(C/4)*3,:,:]
        x4 = x[:,int(C/4)*3:int(C/4)*4,:,:]
        
        sr1 = self.sr1(x1).reshape(B, int(C/4), -1).permute(0, 2, 1)
        sr2 = self.sr2(x2).reshape(B, int(C/4), -1).permute(0, 2, 1)
        sr3 = self.sr3(x3).reshape(B, int(C/4), -1).permute(0, 2, 1)
        sr4 = self.sr4(x4).reshape(B, int(C/4), -1).permute(0, 2, 1)
#         print(sr4.shape)
        sr_h = sr_w = int(sr4.shape[1] ** (1/2))
        sr1 = self.pos1(sr1,sr_h,sr_w)
        sr2 = self.pos2(sr2,sr_h,sr_w)
        sr3 = self.pos3(sr3,sr_h,sr_w)
        sr4 = self.pos4(sr4,sr_h,sr_w)

        x_sr = torch.cat((sr1,sr2,sr3,sr4),dim=2)
        x_sr = self.norm(x_sr)
        kv = self.kv(x_sr).reshape(B, -1, 2, self.num_heads, C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
#         print(q.shape)
#         print(k.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        x = (attn @ v).transpose(1, 2).reshape(B, N, C)
        x = self.proj(x)
        x = self.proj_drop(x)

        return x



class PosCNN(nn.Module):
    def __init__(self, in_chans, embed_dim=768, s=1):
        super(PosCNN, self).__init__()
        self.proj = nn.Conv2d(in_chans, embed_dim, 3, s, 1, bias=True, groups=embed_dim)
        self.s = s

    def forward(self, x, H, W):
        B, N, C = x.shape
        feat_token = x
        cnn_feat = feat_token.transpose(1, 2).view(B, C, H, W)
        if self.s == 1:
            x = self.proj(cnn_feat) + cnn_feat
        else:
            x = self.proj(cnn_feat)
        x = x.flatten(2).transpose(1, 2)
        return x


class TransformerBlock(nn.Module):
    """
    Implementation of Transformer,
    """
    def __init__(self, dim, num_heads, mlp_ratio=4., qkv_bias=False,
                 qk_scale=None, drop=0., attn_drop=0., drop_path=0.,
                 act_layer=nn.GELU, norm_layer=nn.LayerNorm, sr_ratio=2):
        super().__init__()
        self.norm1 = norm_layer(dim)
        self.attn = Attention(dim, num_heads=num_heads, qkv_bias=qkv_bias,
                              qk_scale=qk_scale, attn_drop=attn_drop, proj_drop=drop, sr_ratio=sr_ratio)

        # NOTE: drop path for stochastic depth, we shall see if this is better than dropout here
        self.drop_path = DropPath(drop_path) if drop_path > 0. else nn.Identity()
        self.norm2 = norm_layer(dim)
        mlp_hidden_dim = int(dim * mlp_ratio)
        self.mlp = Mlp(in_features=dim, hidden_features=mlp_hidden_dim, act_layer=act_layer, drop=drop)

    def forward(self, x):
        x = x + self.drop_path(self.attn(self.norm1(x)))
        x = x + self.drop_path(self.mlp(self.norm2(x)))
        return x

class CrossAttention(nn.Module):
    "Implementation of self-attention"

    def __init__(self, dim, num_heads=8, qkv_bias=False,
                 qk_scale=None, attn_drop=0., proj_drop=0.):
        super().__init__()
        self.num_heads = num_heads
        head_dim = dim // num_heads
        self.scale = qk_scale or head_dim ** -0.5

        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, dim * 2, bias=qkv_bias)
        self.attn_drop = nn.Dropout(attn_drop)
        self.proj = nn.Linear(dim, dim)
        self.proj_drop = nn.Dropout(proj_drop)

    def forward(self, x, f_kv,f3):
        x0 = f3
        x_b, x_c, x_h, x_w = x.shape
        x = x.flatten(2).transpose(2,1)
        f_kv = f_kv.flatten(2).transpose(2,1)
        B, N, C = x.shape
        # print(x.shape)
        q = self.q(x).reshape(B, N, self.num_heads, C // self.num_heads).permute(0, 2, 1, 3)

        kv = self.kv(f_kv).reshape(B, N, 2, self.num_heads,
                                  C // self.num_heads).permute(2, 0, 3, 1, 4)
        k, v = kv[0], kv[1]
#         print(q.shape)

        attn = (q @ k.transpose(-2, -1)) * self.scale
        attn = attn.softmax(dim=-1)
        attn = self.attn_drop(attn)

        out = (attn @ v).transpose(1, 2).reshape(B, N, C)
        out = self.proj(out)
        out = self.proj_drop(out)
        out = out.transpose(2,1).reshape(x_b, x_c, x_h, x_w)
        out = out + x0

        return out



def l2_normalization(x):
    norm = torch.norm(x, p=2, dim=1, keepdim=True)
    x = torch.div(x, norm)
    return x


class ResNet34Mvt(nn.Module):
    def __init__(self, num_class=21, dim=768, num_heads=12, mlp_ratio=4., qkv_bias=False, qk_scale=None, attn_drop=0., 
                 drop_path_rate=0.1, drop_rate=0.):
        super(ResNet34Mvt, self).__init__()
        self.backbone = resnet34(pretrained=True)
        self.transformer1 = TransformerBlock(
                     dim=128, num_heads=4,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     attn_drop=attn_drop,
                     drop_path=drop_path_rate)
        self.transformer2 = TransformerBlock(
                     dim=256, num_heads=4,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     attn_drop=attn_drop,
                     drop_path=drop_path_rate)
        self.transformer3 = TransformerBlock(
                     dim=512, num_heads=4,
                     mlp_ratio=mlp_ratio,
                     qkv_bias=qkv_bias,
                     qk_scale=qk_scale,
                     attn_drop=attn_drop,
                     drop_path=drop_path_rate)
        self.cross_attention = CrossAttention(
                     dim=512, num_heads=1, 
                     qkv_bias=False,
                     qk_scale=None, 
                     attn_drop=0., 
                     proj_drop=0.)

        self.pos_drops = nn.Dropout(p=drop_rate)

        self.conv1 = nn.Conv2d(128, 512, kernel_size=1, stride=1)
        self.conv2 = nn.Conv2d(256, 512, kernel_size=1, stride=1)
        
        self.avg1 = nn.AvgPool2d(4,4)
        self.avg2 = nn.AvgPool2d(2,2)

        self.fc = nn.Linear(512, num_class)
        self.fcm = nn.Linear(512, num_class)
        
        self.avgpool = nn.AdaptiveAvgPool2d((1,1))

    def forward(self, x):
        x = self.backbone.conv1(x)
        x = self.backbone.bn1(x)
        x = self.backbone.relu(x)
        x = self.backbone.maxpool(x)

        x = self.backbone.layer1(x)  

        x = self.backbone.layer2(x)  #(32, 128, 28, 28)
        f1_0 = x
        f1_b, f1_c, f1_h, f1_w = x.shape
        f1 = x.flatten(2).transpose(1, 2)
        f1 = self.transformer1(f1)
        f1 = f1.transpose(1, 2)
        f1 = f1.reshape(f1_b, f1_c,f1_h, f1_w)
        f1 = f1 + f1_0

        x = self.backbone.layer3(x)   #(32, 256, 14, 14)
        # print(x.shape)
        f2_0 = x
        f2_b, f2_c, f2_h, f2_w = x.shape
        f2 = x.flatten(2).transpose(1, 2)
        f2 = self.transformer2(f2)
        f2 = f2.transpose(1, 2)
        f2 = f2.reshape(f2_b, f2_c,f2_h, f2_w)
        f2 = f2 + f2_0

        x = self.backbone.layer4(x)

        f3_0 = x
        f3_b, f3_c, f3_h, f3_w = x.shape
        f3 = x.flatten(2).transpose(1, 2)
        f3 = self.transformer3(f3)
        f3 = f3.transpose(1, 2)
        f3 = f3.reshape(f3_b, f3_c,f3_h, f3_w)
        f3 = f3 + f3_0

        f1 = self.conv1(f1)
        f1 = self.avg1(f1)
        
        f2 = self.conv2(f2)
        f2 = self.avg2(f2)
        
        
        fm_cat = self.cross_attention(f1,f2,f3)
        fm = self.avgpool(fm_cat).flatten(1)
        fm = l2_normalization(fm)
        fm_out = self.fcm(fm)
        
        x = self.backbone.avgpool(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        fm = (x + fm_out)/2.

        return fm



