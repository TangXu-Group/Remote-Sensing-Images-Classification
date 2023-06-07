#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
Created on Mon Nov  8 11:16:53 2021

@author: Eric
"""

import argparse
import os
import torch
import torch.nn as nn
from torch.nn import parameter
from torch.utils.data import DataLoader
from torchvision.transforms.functional import scale
from UCM import UCMercedLand
from sampler import CategoriesSampler
from convnet import Convnet, S_Classifier, R_Classifier
from utils import Averager, Timer, count_acc,  euclidean_metric, euclidean_metric2

from torchvision.models import resnet18
from GAM import GAM_Attention

import torchvision.transforms.functional as TF
import torch.nn.functional as F
from DAM import Dynamic_Attention_Module

from cl_data_generator import Dy_Data_Generator
from info_nce import InfoNCE 
from Dynamic_Parameter import DAM
import torch.backends.cudnn as cudnn
import numpy as np
import random
from AMP_Regularizer.amp import AMP
# torch.cuda.set_device(0)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.manual_seed(1)
# SEED = 1

# cudnn.benchmark = False
# cudnn.deterministic = True
# random.seed(SEED)
# np.random.seed(SEED+1)
# torch.manual_seed(SEED+2)
# torch.autograd.set_detect_anomaly(True)
# device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
# print(device)
device_ids = [0,1,2,3]#0, 1 card


def log(path,filename,data):
    best_item = 0.0
    with open(os.path.join(path,filename),'w') as f:
        for item in data:
            f.write("%.4f\n"%item)
            if item >= best_item:
                best_item = item
        f.write("best_acc {:.4f}".format(best_item))
        f.close()

def log_test(path,filename,data):
    best_item = 0.0
    with open(os.path.join(path, filename), 'w') as f:
        for item in data:
            f.write("%.4f\n"%item)
            if item >= best_item:
                best_item = item
        f.write("best_acc {:.4f}".format(best_item))
        f.close()

def cos_dist(x, y):
    n = x.size(0) # query
    m = y.size(0) # shot
    d = x.size(1) # dim
    assert d == y.size(1)

    x_lst = [x[i] for i in range(n)]
    cos_lst = []
    for x_c in x_lst:
        x_c = x_c.unsqueeze(0)
        cos = (F.cosine_similarity(x_c, y, dim=1))*10
        cos_lst.append(cos)
    cosi = torch.cat(cos_lst, dim=0).view(n,m)
    return cosi


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--epochs', type=int, default=40)
    parser.add_argument('--episode', type=int, default=100)
    parser.add_argument('--shot', type=int, default=5)
    parser.add_argument('--query', type=int, default=10)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--numworkers', type=int, default=8)
    parser.add_argument('--lr', type=float, default=1e-4)
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--step_size',type=int, default=5)
    parser.add_argument('--dataset', type=str, default='UC_Merced')
    parser.add_argument('--experiment_time',type=str, default='2')
    parser.add_argument('--metric',type=str,default='euclidean',choices=['euclidean','cosine_similarity'])
    parser.add_argument('--embed_dim',type=int, default=1600)
    parser.add_argument('--trans_num',type=int, default=4)
    parser.add_argument('--dynamic_hyperparameter',type=str, default=False)
    parser.add_argument('--alpha',type=float, default=0.5)
    parser.add_argument('--beta',type=float, default=0.5)
    # parser.add_argument('--ld',type=float, default=0.5)
    # parser.add_argument('--gamma',type=float, default=0.5)
    parser.add_argument('--scale1', type=int, default = 204)
    parser.add_argument('--scale2', type=int, default = 184)
    parser.add_argument('--scale3', type=int, default=164)
    parser.add_argument('--dim',type=int, default=1000)
    parser.add_argument('--dam_dim',type=int, default=4)
    parser.add_argument('--backbone',type=str, default='Convnet')
    parser.add_argument('--reduce_intra_variance',type=bool, default=True)
    
    args = parser.parse_args()
    print(vars(args))

    label_dict = {'UC_Merced': 21, 'NWPU': 45, 'AID': 30}
    label_dim = label_dict[args.dataset]

    log_path={
        'UC_Merced':"/home/Eric/research/MPCL/log/amp/UC_Merced/",
        'AID':"/home/Eric/research/MPCL/log/amp/AID/",
        'NWPU':"/home/Eric/research/MPCL/log/amp/NWPU/"
    }

    csv_path = {
        'UC_Merced':"/home/Eric/research/MPCL/data_csv/UC_Merced/",
        'AID':"/home/Eric/research/MPCL/data_csv/AID/",
        'NWPU':"/home/Eric/research/MPCL/data_csv/NWPU/"
    }

    txt_name=['train_acc_history'+args.experiment_time+'.txt','train_loss_history'+args.experiment_time+'.txt','val_acc_history'+args.experiment_time+'.txt']

    data_path={
        'UC_Merced':"/home/Eric/research/TD-PN-DCNN-main/UC_Merced",
        'AID':"/home/Eric/research/CPT/AID",
        'NWPU':"/home/Eric/research/TD-PN-DCNN-main/NWPU"
        }
    
    data_dir = data_path[args.dataset]
    log_dir = log_path[args.dataset]+str(args.test_way)+'way_'+str(args.shot)+'shot/'
    csv_dir = csv_path[args.dataset]

    trainset = UCMercedLand(data_dir, csv_dir, 'train')
    train_sampler = CategoriesSampler(trainset.label, args.episode,
                                      args.train_way, args.shot+args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers = args.numworkers)
    
    valset = UCMercedLand( data_dir, csv_dir, 'val')
    val_sampler = CategoriesSampler(valset.label, args.episode,
                                    args.test_way, args.shot+args.query)
    val_loader = DataLoader(dataset=valset, batch_sampler=val_sampler,
                              num_workers = args.numworkers)

    
    
    model = Convnet()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    dim1 = args.dim
    dim2 = 600
    scale_classifier = S_Classifier().cuda(device=device_ids[0])
    rot_classifier = R_Classifier().cuda(device=device_ids[0])
    scale_dynamic = DAM(args.embed_dim, args.dam_dim).cuda(device=device_ids[0])
    rot_dynamic = DAM(args.embed_dim, args.dam_dim).cuda(device=device_ids[0])
    # dam = Dynamic_Attention_Module(args.train_way*args.train_way*args.query).to(device)
    # print(args.train_way*args.train_way*args.query)
    

    criterion = nn.CrossEntropyLoss().cuda(device=device_ids[0])
    KDloss = nn.KLDivLoss(reduce=True).cuda(device=device_ids[0])  # logitis_aug = F.log_softmax(logitis_aug, -1)
    MSELoss = nn.MSELoss().cuda(device=device_ids[0])
    contrastive_loss = InfoNCE().cuda(device=device_ids[0])
   
    cl_sample_generator = Dy_Data_Generator().cuda(device=device_ids[0])
    # optimizer = torch.optim.SGD( [{'params': model.parameters()},
    #                                 {'params':scale_classifier.parameters()},
    #                                 {'params':rot_classifier.parameters()},
    #                                 {'params':scale_dynamic.parameters()},
    #                                 {'params':rot_dynamic.parameters()}],
    #                                     lr = 0.01,
    #                                     momentum = 0.9,
    #                                     weight_decay=args.weight_decay)

    optimizer = AMP([{'params': model.parameters()},
                        {'params':scale_classifier.parameters()},
                        {'params':rot_classifier.parameters()},
                        {'params':scale_dynamic.parameters()},
                        {'params':rot_dynamic.parameters()}], 
                        lr=0.01, epsilon=0.5, momentum=0.9)
    
    
    # optimizer = torch.optim.Adam( [{'params': model.parameters()},
    #                                 {'params':scale_classifier.parameters()},
    #                                 {'params':rot_classifier.parameters()},
    #                                 {'params':scale_dynamic.parameters()},
    #                                 {'params':rot_dynamic.parameters()}],
    #                                     lr = args.lr,
    #                                     weight_decay=args.weight_decay)
    # scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=0.2)


    best_acc = 0.0
    train_acc_history = []
    train_loss_history = []
    val_acc_history = []

    # 记录实验设置
    with open(os.path.join(log_dir, 'setting'+args.experiment_time+'.txt'), 'w') as f:
            for k,v in vars(args).items():
                f.write('{}:{}\n'.format(k,v))
            f.close()

    
    timer = Timer()

    for epoch in range(1, args.epochs + 1):
        tl = Averager()
        ta = Averager()
        model.train()
        scale_classifier.train()
        rot_classifier.train()
        rot_dynamic.train()
        scale_dynamic.train()
        acc_train = 0.0
        for i, batch in enumerate(train_loader,1):
            def closure():
                lr = optimizer.state_dict()['param_groups'][0]['lr']
                scale_84, _ = [_.cuda() for _ in batch]
                p = args.shot * args.train_way
                
                scale_104 = TF.resize(scale_84, (args.scale3,args.scale3))
                scale_124 = TF.resize(scale_84, (args.scale2,args.scale2))
                scale_144 = TF.resize(scale_84, (args.scale1,args.scale1))
            
                rot_90 = TF.rotate(scale_84, 90)
                rot_180 = TF.rotate(scale_84, 180)
                rot_270 = TF.rotate(scale_84, 270)
                
                
                # data_shot, data_query = data[:p], data[p:]
                scale_84_fs = model(scale_84)
                scale_104_fs = model(scale_104)
                scale_124_fs = model(scale_124)
                scale_144_fs = model(scale_144)


                rot_0_fs = model(scale_84)
                rot_90_fs = model(rot_90)
                rot_180_fs = model(rot_180)
                rot_270_fs = model(rot_270)

                # scale self-supervision
                shot_84, query_84 = scale_84_fs[:p], scale_84_fs[p:]
                proto_84 = shot_84.reshape(args.shot, args.train_way, -1).mean(dim=0)
                # print(proto_84.size())
                ax = scale_dynamic(proto_84)
                # print(ax.size())
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                # print(p_ld.size())
                # print(n_ld.size())
                anchor_84, p_84, n_84 = cl_sample_generator(proto_84,p_ld,n_ld)
                
                logits_84 = euclidean_metric(query_84, proto_84)
                # nq_logits_84 = euclidean_metric(nq_84, proto_84)
                cl_84 = contrastive_loss(anchor_84, p_84, n_84)

                shot_104, query_104 = scale_104_fs[:p], scale_104_fs[p:]
                proto_104 = shot_104.reshape(args.shot, args.train_way, -1).mean(dim=0)
                ax = scale_dynamic(proto_104)
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                anchor_104, p_104, n_104 = cl_sample_generator(proto_104,p_ld,n_ld)
                # nq_104 = pip_query(query_104,args.gamma)
                # q_104 = cutmix_proto(dim2, query_104)
                logits_104 = euclidean_metric(query_104, proto_104)
                # nq_logits_104 = euclidean_metric(nq_104, proto_104)
                cl_104 = contrastive_loss(anchor_104, p_104, n_104)

                shot_124, query_124 = scale_124_fs[:p], scale_124_fs[p:]
                proto_124 = shot_124.reshape(args.shot, args.train_way, -1).mean(dim=0)
                ax = scale_dynamic(proto_124)
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                anchor_124, p_124, n_124 = cl_sample_generator(proto_124,p_ld,n_ld)
                # nq_124 = pip_query(query_124,args.gamma)
                # q_124 = cutmix_proto(dim2, query_124)
                logits_124 = euclidean_metric(query_124, proto_124)
                # nq_logits_124 = euclidean_metric(nq_124, proto_124)
                cl_124 = contrastive_loss(anchor_124, p_124, n_124)

                shot_144, query_144 = scale_144_fs[:p], scale_144_fs[p:]
                proto_144 = shot_144.reshape(args.shot, args.train_way, -1).mean(dim=0)
                ax = scale_dynamic(proto_144)
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                anchor_144, p_144, n_144 = cl_sample_generator(proto_144,p_ld,n_ld)
                # nq_144 = pip_query(query_144,args.gamma)
                # q_144 = cutmix_proto(dim2, query_144)
                logits_144 = euclidean_metric(query_144, proto_144)
                # nq_logits_144 = euclidean_metric(nq_144, proto_144)
                cl_144 = contrastive_loss(anchor_144, p_144, n_144)

                logits_dict = [logits_84, logits_104, logits_124, logits_144]
                cl_scale_dict = [cl_84, cl_104, cl_124, cl_144]
                # nq_scale_dict = [nq_84, nq_104, nq_124, nq_144]
                scale_fs_dict = [scale_84_fs, scale_104_fs, scale_124_fs, scale_144_fs]
                # logits = torch.cat(logits, 0)
                scale_fs = torch.cat(scale_fs_dict, 0)

                ## 1nd: scale label
                scale_label = torch.arange(args.trans_num, dtype=torch.int8).view(-1, 1).repeat(1, args.query*args.train_way+args.shot*args.train_way).type(torch.LongTensor)
                scale_label = scale_label.view(-1).cuda()
                ## 2nd:  fsl label
                fsl_label = torch.arange(args.train_way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor).cuda()
                ## 3nd: noisy_query label
                # nq_label = fsl_label

                # scale loss
                
                scale_pred = scale_classifier(scale_fs)
                scale_loss = criterion(scale_pred, scale_label)

                # MI mutual information:  KD loss
                raw_logits = sum(logits_dict) / len(logits_dict)
                raw_logits = F.log_softmax(raw_logits, -1)
                MI_losses = [F.kl_div(raw_logits, F.softmax(logits, -1), size_average=True) for logits in logits_dict]
                scale_MI_loss = sum(MI_losses) / len(MI_losses)

                # nq_raw_logits = sum(nq_logits_dict) / len(nq_logits_dict)
                # nq_raw_logits = F.log_softmax(nq_raw_logits, -1)
                # nq_MI_losses = [F.kl_div(nq_raw_logits, F.softmax(logits, -1), size_average=True) for logits in nq_logits_dict]
                # nq_scale_MI_loss = sum(nq_MI_losses) / len(nq_MI_losses)

                # fsl loss for all the tasks copy
                scale_fsl_losses = [F.cross_entropy(logits, fsl_label) for logits in logits_dict]
                scale_fsl_loss = sum(scale_fsl_losses) / len(scale_fsl_losses)

                c_label = torch.arange(args.train_way, dtype=torch.int8).repeat(args.train_way).type(torch.LongTensor).cuda()

                # nq_scale_losses = [F.cross_entropy(logits, fsl_label) for logits in nq_scale_dict]
                # nq_scale_loss = sum(nq_scale_losses) / len(nq_scale_losses)

                #fsl_loss = F.cross_entropy(raw_logits, fsl_label)
                
                acc_list = [count_acc(logits, fsl_label) for logits in logits_dict] # for 4 single angles tasks
                # nq_acc_list = [count_acc(logits, fsl_label) for logits in nq_logits_dict]
                scale_acc = sum(acc_list)/len(acc_list)
                # nq_scale_acc = sum(nq_acc_list)/len(nq_acc_list)
                final_scale_acc = (scale_acc)

                # rotation self-supervision
                shot_0, query_0 = rot_0_fs[:p], rot_0_fs[p:]
                proto_0 = shot_0.reshape(args.shot, args.train_way, -1).mean(dim=0)
                ax = rot_dynamic(proto_0)
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                anchor_0, p_0, n_0 = cl_sample_generator(proto_0,p_ld,n_ld)
                # q_0 = cutmix_proto(dim2, query_0)
                logits_0 = euclidean_metric(query_0, proto_0)
                # nq_logits_0 = euclidean_metric(query_0, proto_0)
                cl_0 = contrastive_loss(anchor_0, p_0, n_0)
                # print(cl_0)

                shot_90, query_90 = rot_90_fs[:p], rot_90_fs[p:]
                proto_90 = shot_90.reshape(args.shot, args.train_way, -1).mean(dim=0)
                ax = rot_dynamic(proto_90)
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                anchor_90, p_90, n_90 = cl_sample_generator(proto_90,p_ld,n_ld)
                # q_90 = cutmix_proto(dim2, query_90)
                logits_90 = euclidean_metric(query_90, proto_90)
                # nq_logits_90 = euclidean_metric(query_90, proto_90)
                cl_90 = contrastive_loss(anchor_90, p_90, n_90)

                shot_180, query_180 = rot_180_fs[:p], rot_180_fs[p:]
                proto_180 = shot_180.reshape(args.shot, args.train_way, -1).mean(dim=0)
                ax = rot_dynamic(proto_180)
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                anchor_180, p_180, n_180 = cl_sample_generator(proto_180,p_ld,n_ld)
                # q_180 = cutmix_proto(dim2, query_180)
                logits_180 = euclidean_metric(query_180, proto_180)
                # nq_logits_180 = euclidean_metric(query_180, proto_180)
                cl_180 = contrastive_loss(anchor_180, p_180, n_180)

                shot_270, query_270 = rot_270_fs[:p], rot_270_fs[p:]
                proto_270 = shot_270.reshape(args.shot, args.train_way, -1).mean(dim=0)
                ax = rot_dynamic(proto_270)
                p_ld = ax[:,0]
                n_ld = ax[:,1]
                anchor_270, p_270, n_270 = cl_sample_generator(proto_270,p_ld,n_ld)
                # q_270 = cutmix_proto(dim2, query_270)
                logits_270 = euclidean_metric(query_270, proto_270)
                # nq_logits_270 = euclidean_metric(query_270, proto_270)
                cl_270 = contrastive_loss(anchor_270, p_270, n_270)

                logits_dict = [logits_0, logits_90, logits_180, logits_270]
                cl_rot_dict = [cl_0, cl_90, cl_180, cl_270]
                # nq_rot_dict = [nq_0, nq_90, nq_180, nq_270]
                rot_fs_dict = [rot_0_fs, rot_90_fs, rot_180_fs, rot_270_fs]
                # logits = torch.cat(logits, 0)
                rot_fs = torch.cat(rot_fs_dict, 0)

                ## 1nd: rotation label
                rot_label = torch.arange(args.trans_num, dtype=torch.int8).view(-1, 1).repeat(1, args.query*args.train_way+args.shot*args.train_way).type(torch.LongTensor)
                rot_label = rot_label.view(-1).cuda()
                ## 2nd:  fsl label
                fsl_label = torch.arange(args.train_way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor).cuda()

                # rotation loss
                
                rot_pred = rot_classifier(rot_fs)
                rot_loss = criterion(rot_pred, rot_label)

                # MI mutual information:  KD loss
                raw_logits = sum(logits_dict) / len(logits_dict)
                raw_logits = F.log_softmax(raw_logits, -1)
                MI_losses = [F.kl_div(raw_logits, F.softmax(logits, -1), size_average=True) for logits in logits_dict]
                rot_MI_loss = sum(MI_losses) / len(MI_losses)

                # contrastive loss
                cl_rot_loss = sum(cl_rot_dict)/len(cl_rot_dict)
                cl_scale_loss = sum(cl_scale_dict)/len(cl_scale_dict)
                # cl_loss = cl_84

                # fsl loss for all the tasks copy
                fsl_losses = [F.cross_entropy(logits, fsl_label) for logits in logits_dict]
                rot_fsl_loss = sum(fsl_losses) / len(fsl_losses)

                # nq_rot_losses = [F.cross_entropy(logits, fsl_label) for logits in nq_rot_dict]
                # nq_rot_loss = sum(nq_rot_losses) / len(nq_rot_losses)
            
                #fsl_loss = F.cross_entropy(raw_logits, fsl_label)

                acc_list = [count_acc(logits, fsl_label) for logits in logits_dict] # for 4 single angles tasks
                # nq_acc_list = [count_acc(logits, fsl_label) for logits in nq_logits_dict]

                rot_acc = sum(acc_list)/len(acc_list)
                # nq_rot_acc = sum(nq_acc_list)/len(nq_acc_list)
                final_rot_acc = rot_acc 

                # print(cl_scale)
                # print(cl_rot)
                # final acc: the average value of two pretext tasks
                final_acc = (final_rot_acc + final_scale_acc)/2
                final_acc_list = []
                final_acc_list.append(final_acc)
                total_loss = 0.5*(scale_fsl_loss + rot_fsl_loss) + args.alpha*(cl_rot_loss+cl_scale_loss)+args.beta*(rot_loss+scale_loss)+0.1*(rot_MI_loss+scale_MI_loss)
                # print(total_loss)

                print('epoch {}, train {}/{}, loss={:.4f} acc={:.4f}'.format(epoch, i, len(train_loader), total_loss.item(), final_acc_list[0]))

                tl.add(total_loss.item())
                ta.add(final_acc_list[0])

                optimizer.zero_grad()
                with torch.autograd.detect_anomaly():
                    total_loss.backward()
                return logits_0, total_loss
            outputs, total_loss = optimizer.step(closure)

        tl = tl.item()
        ta = ta.item()
        
        with torch.no_grad():
            model.eval()
            scale_classifier.eval()
            rot_classifier.eval()
            vl = Averager()
            va = Averager()
        
            for i, batch in enumerate(val_loader, 1):
                scale_84, _ = [_.cuda() for _ in batch]
                p = args.shot * args.test_way
                
                scale_104 = TF.resize(scale_84, (args.scale3,args.scale3))
                scale_124 = TF.resize(scale_84, (args.scale2,args.scale2))
                scale_144 = TF.resize(scale_84, (args.scale1,args.scale1))
            
                rot_90 = TF.rotate(scale_84, 90)
                rot_180 = TF.rotate(scale_84, 180)
                rot_270 = TF.rotate(scale_84, 270)
                
                
                # data_shot, data_query = data[:p], data[p:]
                scale_84_fs = model(scale_84)
                scale_104_fs = model(scale_104)
                scale_124_fs = model(scale_124)
                scale_144_fs = model(scale_144)


                rot_0_fs = model(scale_84)
                rot_90_fs = model(rot_90)
                rot_180_fs = model(rot_180)
                rot_270_fs = model(rot_270)

                # scale self-supervision
                shot_84, query_84 = scale_84_fs[:p], scale_84_fs[p:]
                proto_84 = shot_84.reshape(args.shot, args.train_way, -1).mean(dim=0)
                # nq_84 = cutmix_proto(dim, query_84)
                logits_84 = euclidean_metric(query_84, proto_84)

                shot_104, query_104 = scale_104_fs[:p], scale_104_fs[p:]
                proto_104 = shot_104.reshape(args.shot, args.train_way, -1).mean(dim=0)
                logits_104 = euclidean_metric(query_104, proto_104)

                shot_124, query_124 = scale_124_fs[:p], scale_124_fs[p:]
                proto_124 = shot_124.reshape(args.shot, args.train_way, -1).mean(dim=0)
                logits_124 = euclidean_metric(query_124, proto_124)

                shot_144, query_144 = scale_144_fs[:p], scale_144_fs[p:]
                proto_144 = shot_144.reshape(args.shot, args.train_way, -1).mean(dim=0)
                logits_144 = euclidean_metric(query_144, proto_144)

                logits_dict = [logits_84, logits_104, logits_124, logits_144]
                scale_fs_dict = [scale_84_fs, scale_104_fs, scale_124_fs, scale_144_fs]
                # logits = torch.cat(logits, 0)
                scale_fs = torch.cat(scale_fs_dict, 0)

                ## 1nd: scale label
                scale_label = torch.arange(args.trans_num, dtype=torch.int8).view(-1, 1).repeat(1, args.query*args.train_way+args.shot*args.train_way).type(torch.LongTensor)
                scale_label = scale_label.view(-1).cuda()
                ## 2nd:  fsl label
                fsl_label = torch.arange(args.train_way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor).cuda()

                # scale loss
                
                scale_pred = scale_classifier(scale_fs)
                scale_loss = criterion(scale_pred, scale_label)

                # MI mutual information:  KD loss
                raw_logits = sum(logits_dict) / len(logits_dict)
                raw_logits = F.log_softmax(raw_logits, -1)
                MI_losses = [F.kl_div(raw_logits, F.softmax(logits, -1), size_average=True) for logits in logits_dict]
                scale_MI_loss = sum(MI_losses) / len(MI_losses)

                # fsl loss for all the tasks copy
                scale_fsl_losses = [F.cross_entropy(logits, fsl_label) for logits in logits_dict]
                scale_fsl_loss = sum(scale_fsl_losses) / len(scale_fsl_losses)
                #fsl_loss = F.cross_entropy(raw_logits, fsl_label)
                
                acc_list = [count_acc(logits, fsl_label) for logits in logits_dict] # for 4 single angles tasks
                final_scale_acc = sum(acc_list)/len(acc_list)

                # rotation self-supervision
                shot_0, query_0 = rot_0_fs[:p], rot_0_fs[p:]
                proto_0 = shot_0.reshape(args.shot, args.train_way, -1).mean(dim=0)
                logits_0 = euclidean_metric(query_0, proto_0)

                shot_90, query_90 = rot_90_fs[:p], rot_90_fs[p:]
                proto_90 = shot_90.reshape(args.shot, args.train_way, -1).mean(dim=0)
                logits_90 = euclidean_metric(query_90, proto_90)

                shot_180, query_180 = rot_180_fs[:p], rot_180_fs[p:]
                proto_180 = shot_180.reshape(args.shot, args.train_way, -1).mean(dim=0)
                logits_180 = euclidean_metric(query_180, proto_180)

                shot_270, query_270 = rot_270_fs[:p], rot_270_fs[p:]
                proto_270 = shot_270.reshape(args.shot, args.train_way, -1).mean(dim=0)
                logits_270 = euclidean_metric(query_270, proto_270)

                logits_dict = [logits_0, logits_90, logits_180, logits_270]
                rot_fs_dict = [rot_0_fs, rot_90_fs, rot_180_fs, rot_270_fs]
                # logits = torch.cat(logits, 0)
                rot_fs = torch.cat(rot_fs_dict, 0)

                ## 1nd: rotation label
                rot_label = torch.arange(args.trans_num, dtype=torch.int8).view(-1, 1).repeat(1, args.query*args.train_way+args.shot*args.train_way).type(torch.LongTensor)
                rot_label = rot_label.view(-1).cuda()
                ## 2nd:  fsl label
                fsl_label = torch.arange(args.train_way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor).cuda()

                # rotation loss
                
                # rot_pred = rot_classifier(rot_fs)
                # rot_loss = criterion(rot_pred, rot_label)

                # MI mutual information:  KD loss
                # raw_logits = sum(logits_dict) / len(logits_dict)
                # raw_logits = F.log_softmax(raw_logits, -1)
                # MI_losses = [F.kl_div(raw_logits, F.softmax(logits, -1), size_average=True) for logits in logits_dict]
                # rot_MI_loss = sum(MI_losses) / len(MI_losses)

                # fsl loss for all the tasks copy
                # fsl_losses = [F.cross_entropy(logits, fsl_label) for logits in logits_dict]
                # rot_fsl_loss = sum(fsl_losses) / len(fsl_losses)
                #fsl_loss = F.cross_entropy(raw_logits, fsl_label)
                
                acc_list = [count_acc(logits, fsl_label) for logits in logits_dict] # for 4 single angles tasks
                final_rot_acc = sum(acc_list)/len(acc_list)

                # final acc: the average value of two pretext tasks
                final_acc = (final_rot_acc + final_scale_acc)/2
                final_acc_list = []
                final_acc_list.append(final_acc)
                # total_loss = 0.5*(scale_fsl_loss + rot_fsl_loss) +args.alpha*(scale_loss+scale_MI_loss+rot_loss+rot_MI_loss)
                vl.add(total_loss.item())
                va.add(final_acc_list[0])
                
                

            vl = vl.item()
            va = va.item()
            # scheduler.step()
            if va > best_acc:
                best_acc = va
                # state = {'model':model.state_dict(),'dam':dam.state_dict()}
                state = {'model':model.state_dict(),'scale_classifier':scale_classifier.state_dict(),'rot_classifier':rot_classifier.state_dict()}
                torch.save(state, '/home/Eric/research/MPCL/checkpoint/'+args.dataset+'/'+str(args.test_way)+'_way'+str(args.shot)+'_shot'+'_model'+args.experiment_time+'.pth')
            print('epoch {}, val, loss={:.4f} acc={:.4f} best_acc={:.4f}'.format(epoch, vl, va, best_acc))

            
            
            print('ETA:{}/{}'.format(timer.measure(),timer.measure(epoch/args.epochs)))
            train_acc_history.append(ta)
            train_loss_history.append(tl)
            val_acc_history.append(va)
            log(log_dir,txt_name[0],train_acc_history)
            log(log_dir,txt_name[1],train_loss_history)
            log(log_dir,txt_name[2],val_acc_history)
