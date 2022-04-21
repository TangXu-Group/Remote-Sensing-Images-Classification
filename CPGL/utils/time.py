import os
import shutil
import time
import pprint

import torch
# import numpy as np
import torch.nn as nn


def set_gpu(x):
    os.environ['CUDA_VISIBLE_DEVICES'] = x
    print('using gpu:', x)


def ensure_path(path):
    if os.path.exists(path):
        if input('{} exists, remove? ([y]/n)'.format(path)) != 'n':
            shutil.rmtree(path)
            os.makedirs(path)
    else:
        os.makedirs(path)


class Averager():

    def __init__(self):
        self.n = 0
        self.v = 0

    def add(self, x):
        self.v = (self.v * self.n + x) / (self.n + 1)
        self.n += 1

    def item(self):
        return self.v


def count_acc(logits, label):
    pred = torch.argmax(logits, dim=1)
    return (pred == label).type(torch.cuda.FloatTensor).mean().item()


def dot_metric(a, b):
    return torch.mm(a, b.t())


def euclidean_metric(a, b):
    n = a.shape[0]#n=40
    m = b.shape[0]#m=10
    """
    a.unsqueeze(1)的size为[40,1,4096]
    b.unsqueeze(0)的size为[1,10,4096]
    a = a.unsqueeze(1).expand(n, m, -1)的size为[40,10,4096]
    b = b.unsqueeze(0).expand(n, m, -1)的size为[40,10,4096]
    logits的size为[40,10]
    """
    a = a.unsqueeze(1).expand(n, m, -1)
    b = b.unsqueeze(0).expand(n, m, -1)
    
    logits = -((a - b)**2).sum(dim=2)
    return logits



# class MahalanobisLayer(nn.Module):

#     def __init__(self, dim1,dim2,dim3,decay = 0.1):
#         super(MahalanobisLayer, self).__init__()
#         self.register_buffer('S', torch.ones(dim1,dim2,dim3))
#         self.register_buffer('S_inv', torch.ones(dim1,dim2,dim3))
#         self.decay = decay

#     def forward(self, x, x_fit):
#         """
#         Calculates the squared Mahalanobis distance between x and x_fit
#         """

#         delta = x - x_fit
#         for i in range(4096):
#             print(self.S[:,:,i].size())
#             print(self.cov(delta[:,:,i]).size())
#             self.S[:,:,i] = (1 - self.decay)*self.S[:,:,i] + (self.decay )*self.cov(delta[:,:,i])
#             self.S_inv[:,:,i] = torch.pinverse(self.S[:,:,i])
#         # print(delta.size())
#         # print(self.S_inv.size())
#         # m = torch.mul(torch.mm(delta, self.S_inv), delta)#m的size为40✖40，行为class重复4次，列为40个query
       
#         m = torch.mul(torch.mul(delta,self.S_inv),delta)
#         # return torch.diag(m)
#         # m = m[:10,:].t()
#         m = -m.sum(dim=2)
#         return m

    def cov(self, x):
        x -= torch.mean(x, dim=0)
        return 1 / (x.size(0) - 1) * x.t().mm(x)

    def update(self, X, X_fit):
        delta = X - X_fit
        for i in range(4096):
            self.S[:,:,i] = (1 - self.decay) * self.S[:,:,i] + self.decay * self.cov(delta[:,:,i])
            self.S_inv[:,:,i] = torch.pinverse(self.S[:,:,i])

class Timer():

    def __init__(self):
        self.o = time.perf_counter()

    def measure(self, p=1):
        x = (time.perf_counter() - self.o) / p
        x = int(x)
        if x >= 3600:
            return '{:.1f}h'.format(x / 3600)
        if x >= 60:
            return '{}m'.format(round(x / 60))
        return '{}s'.format(x)

_utils_pp = pprint.PrettyPrinter()
def pprint(x):
    _utils_pp.pprint(x)


def l2_loss(pred, label):
    return ((pred - label)**2).sum() / len(pred) / 2

