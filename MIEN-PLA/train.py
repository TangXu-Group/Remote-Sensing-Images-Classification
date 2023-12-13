
import random
import numpy as np
from math import sqrt
from datetime import * 

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models
from torch.utils.data import DataLoader
from core import *
from builder import *
from ema import EMA
from model_Transformer import MIEN


train_root =  '/root/autodl-tmp/dataset/NWPU/train/'
test_root = '/root/autodl-tmp/dataset/NWPU/test/'


def CLloss(logits, cl_labels):
    """
    :param logits: shape: (N, C)
    :param cl_labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert cl_labels.size(0) == N and cl_labels.size(1) == C, f'label tensor shape is {cl_labels.shape}, while logits tensor shape is {logits.shape}'

    softmax_logits = F.softmax(logits,dim=1)
    log_softmax_logits = torch.log(1-softmax_logits)
    losses = -torch.sum(log_softmax_logits * cl_labels, dim=1)  # (N)
    return losses

def cross_entropy(logits, labels):
    """
    :param logits: shape: (N, C)
    :param labels: shape: (N, C)
    :param reduction: options: "none", "mean", "sum"
    :return: loss or losses
    """
    N, C = logits.shape
    assert labels.size(0) == N and labels.size(1) == C, f'label tensor shape is {labels.shape}, while logits tensor shape is {logits.shape}'

    log_logits = F.log_softmax(logits, dim=1)
    losses = -torch.sum(log_logits * labels, dim=1)  # (N)


    return losses


def generate_compl_labels(label,po_label,dev):
    K = 45
    candidates = np.arange(K)
    candidates = np.repeat(candidates.reshape(1, K), len(label), 0)
    mask = np.ones((len(label), K), dtype=bool)
    mask[range(len(label)), label.cpu().numpy()] = False
    mask[range(len(label)), po_label.cpu().numpy()] = False

    # candidates_ = candidates[mask]#.reshape(len(label), K-1)
    candidates_ = [candidates[i][mask[i]] for i in range(len(label))]
    # idx = np.random.randint(0, K-1, len(label))
    # complementary_labels = candidates_[np.arange(len(label)), np.array(idx)]
    complementary_labels =[]
    for i in range(len(label)):
        idx = np.random.randint(0, len(candidates_[i]))
        complementary_labels.append(candidates_[i][idx])
    return torch.tensor(complementary_labels).to(dev)


def create_models(nb_class,dev,h,numblocks):
    net = MIEN(num_class=nb_class,h=h,numblocks=numblocks)
    net_ema = MIEN(num_class=nb_class,h=h,numblocks=numblocks)

    for name,param in net_ema.named_parameters():
        param.requires_grad = False

    return net.to(dev),net_ema.to(dev)


def create_loader(train_root,test_root,noise_rate,noise_type):
    transform = build_transform(rescale_size=224,crop_size=224)
    dataset = build_hrrs_dataset(train_root,test_root ,transform['train'], transform['test'],noise_rate,n=45,noise_type=noise_type)
    train_loader = DataLoader(
            dataset['train'], batch_size=64, shuffle=True,num_workers=16
            )
    print(len(train_loader.dataset))
    test_loader = DataLoader(
            dataset['test'], batch_size=64, shuffle=False,num_workers=16
            )
    print(len(test_loader.dataset))

    return train_loader,test_loader
def get_pred_acc(pred, y):
    N,C = pred.shape
    softmax_acc = F.softmax(pred,dim=1)
    pred_acc = []
    for i in range(N):
        pred_acc.append(softmax_acc[i][y[i]].data)
    pred_acc = torch.tensor([acc for acc in pred_acc])
    return pred_acc
import argparse
parser = argparse.ArgumentParser(description='PyTorchUCM Training')
parser.add_argument('--noise_rate', default=0.1,type=float,required=True)
parser.add_argument('--noise_type', default='symmetric',required=False)
parser.add_argument('--dev', default=0,type=int,required=False)
parser.add_argument('--h', default=4,type=int,required=False)
parser.add_argument('--nblock', default=1,type=int,required=False)
args = parser.parse_args()

dev = torch.device(f"cuda:0" if torch.cuda.is_available() else "cpu")
random.seed(123)
torch.manual_seed(123)
torch.cuda.manual_seed_all(123)
print(args)
error_cl=0
error_ol=0

net,net_ema = create_models(45,dev,args.h,args.nblock)
ema = EMA(net, alpha=0.99)
ema.apply_shadow(net_ema)
optimizer = torch.optim.SGD(net.parameters(), lr=0.005, momentum=0.9)


noise_rate = args.noise_rate
noise_type = args.noise_type
train_loader,test_loader = create_loader(train_root,test_root,noise_rate,noise_type)
time_qishi = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
epoch1=50
epoch2=50
epoch3=50


print('------Start training------')
print(f'------{time_qishi}------')
time_1 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
for epoch in range(2+epoch1+epoch2+epoch3):
    net.train()
    for batch_idx,sample in enumerate(train_loader):
        data = sample['data1'].to(dev)
        labels = sample['noise_label'].long().to(dev)
        true_labels = sample['true_label']
        output = net(data)
        N, C = output.shape
        given_labels = torch.full(size=(N, C), fill_value=0.1/(C - 1)).to(dev)
        given_labels.scatter_(dim=1, index=torch.unsqueeze(labels, dim=1), value=1-0.1)    
        with torch.no_grad():
            output_ema = net_ema(data)
        if epoch < 2:
            losses = cross_entropy(output,given_labels)
            loss = losses.mean()
        elif epoch<(2+epoch1):
            pre_labels= output_ema.data.max(1, keepdim=True)[1] 
            po_labels = torch.tensor([po_label for po_label in pre_labels]).to(dev)
            cl_labels = generate_compl_labels(labels,po_labels,dev)
            made_cl_labels = torch.full(size=(N, C), fill_value=0.1/(C - 1)).to(dev)
            made_cl_labels.scatter_(dim=1, index=torch.unsqueeze(cl_labels, dim=1), value=1-0.1)
            losses = CLloss(output,made_cl_labels)
            loss = losses.mean()
        elif epoch<(2+epoch1+epoch2):
            hard_labels = output.max(1, keepdim=False)[1]
            hard_labels_ema  = output_ema.max(1, keepdim=False)[1]
            losses = cross_entropy(output,given_labels)
            loss = losses[(hard_labels ==hard_labels_ema)*(hard_labels ==labels)].mean()
        else:
            soft_labels_ema = F.softmax(output_ema, dim=1) 
            soft_labels = F.softmax(output, dim=1) 
            pred_acc = get_pred_acc(output, labels).to(dev)
            losses1 = cross_entropy(output, given_labels) 
            losses2 = cross_entropy(output, soft_labels_ema)
            losses = losses1*pred_acc+losses2*(1-pred_acc)
            loss = losses.mean()

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        ema.update_params(net)
        ema.apply_shadow(net_ema)
    if epoch ==(2+epoch1+epoch2+epoch3-15-1):
        for group in optimizer.param_groups:
            group['lr'] = group['lr']*0.5    
    elif epoch ==(2+epoch1+epoch2+epoch3-10-1) or  epoch ==(2+epoch1+epoch2+epoch3-1-5):
        for group in optimizer.param_groups:
            group['lr'] = group['lr']*0.5   
    if epoch==1:
        print('------Warmup Endding------')
        acc1 = evaluate(test_loader, net, dev, topk=(1,))
        acc1 = format(acc1, '.2f')
        print(f'Warmup acc1  :{acc1}')
    elif epoch==(2+epoch1-1):
        print('------CLtrain Endding------')
        acc2 = evaluate(test_loader, net, dev, topk=(1,))
        acc2 = format(acc2, '.2f')
        print(f'CLtrain acc2  :{acc2}')
    elif epoch==(2+epoch1+epoch2-1):
        print('------OLtrain1 Endding------')
        acc3 = evaluate(test_loader, net, dev, topk=(1,))
        acc3 = format(acc3, '.2f')
        print(f'OLtrain acc3  :{acc3}')

time_2 = datetime.now().strftime('%Y-%m-%d %H:%M:%S')
time_end = datetime.now().strftime('%Y-%m-%d %H:%M:%S')

print(f'------{time_end}------')
print('------Start testing------')
acc = evaluate(test_loader, net, dev, topk=(1,))
print('end acc:', format(acc, '.2f'))