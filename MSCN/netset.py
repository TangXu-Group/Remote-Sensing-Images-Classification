import torch

import torch.nn as nn
import torchvision.models as models
import network.MSCN

base_dir = '/home/amax/JW/'

def net_set(args):
    if args.model == 'MSCN':
        model = network.MSCN.MSCN(args.nb_classes,args.res,args.k1,args.k2)
        ckpt_state_dict = torch.load(base_dir + f'pretrain/resnet{args.res}.pth')
        model.resnet.load_state_dict(ckpt_state_dict)
        ckpt_state_dict = None
    return model,ckpt_state_dict
