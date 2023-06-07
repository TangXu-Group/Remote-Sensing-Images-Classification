import torch
import torch.nn as nn
import torch.nn.functional as F

class DAM(nn.Module):
    def __init__(self, dim, reduction=4, K=2, t=30):
        super(DAM,self).__init__()
        self.t = t
        self.ADM = nn.Sequential(
            nn.Linear(dim, dim // reduction, bias=False),
            nn.ReLU(inplace=True),
            nn.Linear(dim // reduction, K, bias=False),
        )

        self.ADM2 = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, K, kernel_size=1),
        )

        self.ADM3 = nn.Sequential(
            nn.Conv2d(dim, dim // reduction, kernel_size=1, groups=4),
            nn.BatchNorm2d(dim // reduction),
            nn.ReLU(inplace=True),
            nn.Conv2d(dim // reduction, K, kernel_size=1),
        )

    def forward(self,x):
        x = x.unsqueeze(2).unsqueeze(3)
        y = self.ADM3(x)
        y = torch.flatten(y,1)
        ax = F.softmax(y, dim = 1)
        return ax


if __name__ == '__main__':
    proto = torch.randn([5,1600]).cuda()
    dp = DAM(proto.size(1)).cuda()
    a = dp(proto)
    # p = a[:,0]
    # n = a[:,1]
    # cl_data = Data_Generator2(p_ld=p,n_ld=n).cuda()
    # anchor_query,positive_query,negative_query = cl_data(proto)
    print(a[:,0])
    print(a[:,1])

    