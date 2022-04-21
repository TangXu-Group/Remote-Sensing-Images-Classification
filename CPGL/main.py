#!/usr/bin/env python3
# -*- coding: utf-8 -*-
import argparse
import os
import os.path as osp
from pickle import FALSE
import torch
import torch.nn.functional
from torch.utils.data import DataLoader
from torchvision import transforms,datasets
# from ucmercedland import UCMercedLand
from dataset.dataloader import ImageLabelLoader
from dataset.samplers import CategoriesSampler
# from convnet import FEM,softmax,resnet18,s_auxiliary_softmax,v_auxiliary_softmax
import torch.nn as nn
import random
import numpy as np
import torchvision.transforms.functional as F
import matplotlib.pyplot  as plt
from models.modules import resnet18,softmax,Averager
from utils.time import Timer
# from TopKForEachClass import top_k_per_class,refined_proto_calculate
from models.TopkQueryToShot import top_k_per_class,refined_proto_calculate
from models.feature_clustering import Feature_Clustering
from utils.plot_confusion_matrix import plotconfusion


from utils.earlystopping import EarlyStopping

torch.cuda.set_device(0)
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.manual_seed(1)
device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)

def confidence_penalty_loss(y_pred, y_true, beta=0.1):
    epsilon = 1e-5
    entropy = -torch.mean(torch.sum(y_pred*torch.log(y_pred) + epsilon),-1)
    penalty = torch.nn.functional.cross_entropy(y_pred, y_true)
    return penalty - beta*entropy

def log_rewrite(path,filename):
    with open(os.path.join(path,filename),'r') as f:
        previous_data = f.readline()
        f.close()
    return previous_data

def count_acc(logits,labels):
    pred = torch.argmax(logits,1)
    gt = labels
    return (pred == gt).type(torch.cuda.FloatTensor).mean()


def log(path,filename,data):
    with open(os.path.join(path,filename),'w') as f:
        for item in data:
            f.write("%s\n"%item)
        f.close()

def log_test(path,filename,data):
    best_item = 0.0
    with open(os.path.join(path,filename),'w') as f:
        for item in data:
            f.write("%.4f\n"%item)
            if item >= best_item:
                best_item = item
        f.write("best_acc {:.4f}".format(best_item))
        f.close()

def image_scale(s,d,images):
    s_max = s + 10*d
    s_min = s - 10*d
    gap = 13
    choice = np.linspace(s_min, s_max, gap)
    index = [6]
    new_choice = np.delete(choice, index)
    final_choice = random.sample(list(new_choice),1)
    final_scale = int(final_choice[0])
    scaled_image = F.resize(images,(final_scale,final_scale))
    return scaled_image,final_scale


if __name__ == '__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--feature_clustering', type=str, default=True)
    # parser.add_argument('--feature_clustering2', type=str, default=True)
    parser.add_argument('--epoch', type=int, default=40)
    parser.add_argument('--change_iteration', default=4, type=int)
    parser.add_argument('--k', type=int, default=2, help="top k confident embeded features of the query set")
    parser.add_argument('--reload_flag', default=False, type=str)
    parser.add_argument('--shot', type=int, default=2)
    parser.add_argument('--query', type=int, default=3)
    parser.add_argument('--train_way', type=int, default=5)
    parser.add_argument('--gpu', default='0')
    parser.add_argument('--batchsize', type=int, default=128)
    parser.add_argument('--numworkers', type=int, default=8)
    parser.add_argument('--episode', type=int, default=100)
    parser.add_argument('--lr', type=float, default=0.0001)
    parser.add_argument('--stepsize', type=int, default=20)
    parser.add_argument('--feature_dims', type=int, default=512)
    parser.add_argument('--weight_decay', type=float, default=0.0005)
    parser.add_argument('--dataset', default='AID', type=str)
    parser.add_argument('--data_num', default='5', type=str)
    parser.add_argument('--experiment_time', default='6', type=str)
    parser.add_argument('--s', default=224, type=int)
    parser.add_argument('--d', default=6, type=int)
    parser.add_argument('--topk_query_for_a_shot', default=True)
    parser.add_argument('--alpha', default=0.8, type=float)
    parser.add_argument('--beta', default=2.0, type=float)
    parser.add_argument('--save_model', default=False, type=bool)
    parser.add_argument('--confusion_matrix', default=False, type=bool)
    parser.add_argument('--test_scale', default=224, type=int)
    args = parser.parse_args()
    
    print(vars(args))
    
    
    label_dict = {'UC_Merced': 21, 'NWPU': 45, 'AID': 30}
    label_dim = label_dict[args.dataset]

    log_path={
        'UC_Merced':"/home/Eric/research/TD-PN-DCNN-main/log/time_cost/UC_Merced/feature_clustering/"+args.data_num,
        'AID':"/home/Eric/research/TD-PN-DCNN-main/log/time_cost/AID/feature_clustering/"+args.data_num,
        'NWPU':"/home/Eric/research/TD-PN-DCNN-main/log/time_cost/NWPU/feature_clustering/"+args.data_num
    }

    csv_path = {
        'UC_Merced':"/home/Eric/research/TD-PN-DCNN-main/data_csv/UC_Merced/",
        'AID':"/home/Eric/research/TD-PN-DCNN-main/data_csv/AID/",
        'NWPU':"/home/Eric/research/TD-PN-DCNN-main/data_csv/NWPU/"
    }

    txt_name=['train_acc_history'+args.experiment_time+'.txt','train_loss_history'+args.experiment_time+'.txt','test_acc_history'+args.experiment_time+'.txt', 'best_scale'+args.experiment_time+'.txt']

    data_path={
        'UC_Merced':"/home/Eric/research/TD-PN-DCNN-main/UC_Merced",
        # 'UC_Merced':"/home/lin/tgars2021_comparecode/DCNN/dataset/UC_Merced",
        'AID':"/home/Eric/research/TD-PN-DCNN-main/AID",
        'NWPU':"/home/Eric/research/TD-PN-DCNN-main/NWPU"
        }
    
    kernel_num_dir={
        'UC_Merced':2048,
        'AID':4096,
        'NWPU':3072
        }
    
    data_dir = data_path[args.dataset]
    log_dir = log_path[args.dataset]
    csv_dir = csv_path[args.dataset]
    # kernel_num = kernel_num_dir[args.dataset]
    softmax_input =  512
    # print("The number of 1X1 convolutional kernel is:{}".format(kernel_num))
    
    
    trainset = ImageLabelLoader('train'+args.data_num,data_dir,csv_dir, 'train')
    # trainset = UCMercedLand('5%/train',data_dir,'train')
    train_sampler = CategoriesSampler(trainset.label, args.episode,
                                      args.train_way, args.shot + args.query)
    train_loader = DataLoader(dataset=trainset, batch_sampler=train_sampler,
                              num_workers=args.numworkers)
    """
    获取测试数据的路径
    """   
    testset = ImageLabelLoader('test'+args.data_num,data_dir,csv_dir, 'test')
    test_loader = DataLoader(testset,
                            batch_size=args.batchsize,
                            shuffle=True,
                            num_workers=8)
    
  
    test_num = len(testset)

    """全局fine-tune"""   
    snet = resnet18()
    vnet = resnet18()
    
     # net = load_pretrained_weight(net).to(device)
    model_weight_path = '../DCNN/resnet18-5c106cde.pth'
    snet.load_state_dict(torch.load(model_weight_path),strict=False)
    inchannel = snet.fc.in_features
    snet.fc = nn.Linear(512,args.feature_dims)
    snet.to(device)

    vnet.load_state_dict(torch.load(model_weight_path),strict=False)
    inchannel = vnet.fc.in_features
    vnet.fc = nn.Linear(512,args.feature_dims)
    vnet.to(device)
    
    
    
    classifier = softmax(input_dim=softmax_input,num_classes=label_dim).to(device)
    
    criterion = nn.CrossEntropyLoss().to(device)
    
    # d_loss = distance_loss().to(device)
    
       
    optimizer = torch.optim.Adam([{'params':snet.parameters()},{'params':vnet.parameters()},{'params':classifier.parameters()}],
                                 lr=args.lr,
                                 weight_decay = args.weight_decay)
    
    scheduler = torch.optim.lr_scheduler.StepLR(optimizer, step_size=args.stepsize, gamma=0.1)
    
    best_acc = 0.0
    train_acc_history = []
    train_loss_history = []
    test_acc_history = []
    
     #如果reload_flag = True,则加载已保存的模型继续训练，刚开始reload_flag = False
    if args.reload_flag:
        previous_train_acc = log_rewrite(log_dir,txt_name[0])
        previous_train_loss = log_rewrite(log_dir,txt_name[1])
        previous_test_acc = log_rewrite(log_dir,txt_name[2])
        train_acc_history.append(previous_train_acc)
        train_loss_history.append(previous_train_loss)
        test_acc_history.append(previous_test_acc)
        checkpoint1 = torch.load('./checkpoint/UC_Merced/feature_clustering/'+args.data_num+'/snet.pth')
        checkpoint2 = torch.load('./checkpoint/UC_Merced/feature_clustering/'+args.data_num+'/vnet.pth')
        checkpoint3 = torch.load('./checkpoint/UC_Merced/feature_clustering/'+args.data_num+'/classifier.pth')
        snet.load_state_dict(checkpoint1['model'])
        optimizer.load_state_dict(checkpoint1['optimizer'])
        start_epoch = checkpoint1['epoch']
        record_scale = checkpoint1['record_scale']
        vnet.load_state_dict(checkpoint2['model'])
        classifier.load_state_dict(checkpoint3['model'])
        print('加载 epoch {} 成功！'.format(start_epoch))
    else:
        start_epoch = 1
        print('无保存模型，将从头开始训练！')

    # 记录实验设置
    with open(os.path.join(log_dir, 'setting'+args.experiment_time+'.txt'), 'w') as f:
            for k,v in vars(args).items():
                f.write('{}:{}\n'.format(k,v))
            f.close()

    # loss_list = []
    # alpha = 0.01
    timer = Timer()
    testlabels = []
    predicts=[]
    best_predict=[]
    best_label = []
    early_stopping = EarlyStopping(patience=15, verbose=True)
    for epoch in range(start_epoch, args.epoch + 1):
        tl = Averager()
        ta = Averager()
        snet.train()
        vnet.train()
        classifier.train()
        acc_train = 0.0 
        for i, batch in enumerate(train_loader, 1):#batch就是episode
            lr = optimizer.state_dict()['param_groups'][0]['lr']
            train_num = len(train_loader)
            s_data,true_label = [_.cuda() for _ in batch]
            
            #change scale and alpha 
            if epoch==1 or epoch % args.change_iteration == 0:
                v_data,vscale = image_scale(args.s,args.d,s_data)
                record_scale = vscale
                
            else :
                v_data = F.resize(s_data,(record_scale,record_scale))
                
            # if epoch % args.alpha_iteration == 0:
            #     alpha = alpha * 0.5
           
            p = args.shot*args.train_way

            # label = torch.arange(args.train_way).repeat(args.query)
            
            # label = label.type(torch.cuda.LongTensor)
            s_labels = np.tile(np.arange(args.train_way)[:, np.newaxis],(1,args.shot)).astype(np.uint8).transpose(1,0)
            q_labels = np.tile(np.arange(args.train_way)[:, np.newaxis],(1,args.query)).astype(np.uint8).transpose(1,0)

            s_labels = torch.from_numpy(np.reshape(s_labels,(-1,)))
            q_labels = torch.from_numpy(np.reshape(q_labels,(-1,)))

            s_labels = s_labels.type(torch.LongTensor)
            q_labels = q_labels.type(torch.LongTensor)

            s_onehot = torch.zeros(args.train_way*args.shot, args.train_way).scatter_(1, s_labels.view(-1,1), 1)
            q_onehot = torch.zeros(args.train_way*args.query, args.train_way).scatter_(1, q_labels.view(-1,1), 1)
            #the few-shot learning of the original feature
            #s_data
            s_support, s_query = s_data[:p], s_data[p:]
            
            s_support_emb = snet(s_support)
            
            s_query_emb = snet(s_query)
            
            
            #v_data
            v_support, v_query = v_data[:p], v_data[p:]

            v_support_emb = vnet(v_support)
            
            v_query_emb = vnet(v_query)

            # multiscale data
            multiscale_support_emb = (s_support_emb + v_support_emb)/2 # (N_way*N_shot)x512
           
            multiscale_query_emb = (s_query_emb + v_query_emb)/2 # (N_way*N_query)x512

            all_multiscale_emb = torch.cat((multiscale_support_emb,multiscale_query_emb),dim=0)
            # print(all_multiscale_emb.size())
            """
                calculating the prototype of each class and the assignment A
                multiscale_support_emb: (N_way*N_shot)x512
                multiscale_query_emb:   (N_way*N_query)x512
                s_onehot:               (N_way*N_shot)xN_way, one-hot
                q_onehot:               (N_way*N_query)xN_way, one-hot
            """
            multiscale_prototype = multiscale_support_emb.reshape(args.shot,args.train_way,-1).mean(dim=0)# a (N_way,512) Tensor

            per_mean = all_multiscale_emb.view(args.shot+args.query,args.train_way,-1).mean(dim=0)
            all_mean = all_multiscale_emb.mean(dim=0)# a (1,512) Tensor
            R_fc1 = Feature_Clustering(all_multiscale_emb, per_mean, all_mean)
            
          
            multiscale_query_emb = multiscale_query_emb.view(-1,args.feature_dims)
            n = multiscale_query_emb.shape[0]
            m = multiscale_prototype.shape[0]
            a = multiscale_query_emb.unsqueeze(1).expand(n,m,-1)
            b = multiscale_prototype.unsqueeze(0).expand(n,m,-1)
            dist = ((a-b)**2).mean(dim=2)# a (query*train_way, train_way) tensor

            #top-k selection and calulating the refined prototype
            refined_multiscale_support_emb = top_k_per_class(multiscale_query_emb,multiscale_support_emb,dist,args.k)
            refined_shot = int(refined_multiscale_support_emb.shape[0]/args.train_way)
            refined_prototype,proto_acc,refined_dist = refined_proto_calculate(refined_multiscale_support_emb,multiscale_query_emb,refined_shot,q_onehot,args.train_way)

            

            #calculating loss
           
            loss_proto = criterion(-dist.cuda(), torch.argmax(q_onehot.cuda(),1))
            loss_refined_proto = criterion(-refined_dist.cuda(), torch.argmax(q_onehot.cuda(),1))
            multiscale_support_logits = classifier(multiscale_support_emb)
            multiscale_query_logits = classifier(multiscale_query_emb)
            # print(multiscale_support_logits)
            # print(multiscale_query_logits)
            loss_shot = criterion(multiscale_support_logits.cuda(),true_label[:p])
            loss_query = criterion(multiscale_query_logits.cuda(),true_label[p:])
            # print(loss_shot)
            # print(loss_query)
            loss_total = args.alpha*(loss_proto + loss_refined_proto) + loss_shot + loss_query + args.beta*(R_fc1)
            # loss_list.append(loss_total)

            classification_acc = count_acc(torch.cat((multiscale_support_logits.cuda(),multiscale_query_logits.cuda()),0), true_label)
            # enchanced_acc = count_acc(enchanced_logits_proto, label)
            print('epoch {}, train {}/{}, lr={:.7f}, loss_total={:.4f},loss_proto={:.4f},loss_refined_proto={:.4f},R_fc1={:.4f},cf_acc={:.4f},proto_acc={:.4f}'
                  .format(epoch, i, len(train_loader),lr, loss_total,loss_proto.item(),loss_refined_proto.item(), R_fc1,classification_acc.item(),proto_acc.item()))
            
            
            tl.add(loss_total.item())
            ta.add(classification_acc.item())
            optimizer.zero_grad()
            
            loss_total.backward()
            
            optimizer.step()
        
        tl = tl.item()
        ta = ta.item()
        train_acc_history.append(ta)
        train_loss_history.append(tl)
        # test_acc_history.append(va)
        log(log_dir,txt_name[0],train_acc_history)
        log(log_dir,txt_name[1],train_loss_history)
        scheduler.step()
        

        snet.eval()
        vnet.eval()
        classifier.eval()
        vl = Averager()
        va = Averager()
        
        with torch.no_grad():
            acc = 0.0
            for test_data in test_loader:
               
                test_images, test_labels = test_data
                scaled_test_image = F.resize(test_images,(args.test_scale,args.test_scale))
                # scaled_test_image = F.resize(test_images,(record_scale,record_scale))
                
               
                soutput = snet(test_images.cuda())
                voutput = vnet(scaled_test_image.cuda())
                output = (soutput+voutput)/2
               
                logits = classifier(output)
               
                predict = torch.max(logits, dim=1)[1]
               
               
                acc += (predict == test_labels.cuda()).sum().item()
                
                testlabels.append(test_labels.cpu().numpy())
                predicts.append(predict.cpu().numpy())
            test_acc = acc / test_num
            test_acc_history.append(test_acc)
            
            if test_acc>best_acc:
                best_acc = test_acc
                best_predict = predicts
                best_label = testlabels
                if args.save_model:
                    state1 = {'model':snet.state_dict(),'optimizer':optimizer.state_dict(), 'epoch':epoch, 'record_scale':record_scale}
                    torch.save(state1, '/home/Eric/research/TD-PN-DCNN-main/checkpoint/'+args.dataset+'/feature_clustering/'+args.data_num+'/snet'+args.experiment_time+'.pth')
                    state2 = {'model':vnet.state_dict()}
                    torch.save(state2, '/home/Eric/research/TD-PN-DCNN-main/checkpoint/'+args.dataset+'/feature_clustering/'+args.data_num+'/vnet'+args.experiment_time+'.pth')
                    state3 = {'model':classifier.state_dict()}
                    torch.save(state3, '/home/Eric/research/TD-PN-DCNN-main/checkpoint/'+args.dataset+'/feature_clustering/'+args.data_num+'/classifier'+args.experiment_time+'.pth')
                
            print('test_accuracy: %.4f  best_accuracy:%.4f' %(test_acc,best_acc))
            print('ETA:{}/{}'.format(timer.measure(), timer.measure(epoch / args.epoch)))
        
            
            log_test(log_dir,txt_name[2],test_acc_history)
           

            early_stopping(test_acc)
            if early_stopping.early_stop:
                if args.confusion_matrix:
                    print('plotting confusion matrix')
                    plotconfusion(best_label,best_predict,log_dir)
                print("Early stopping!")
                break
        # print(testlabels)
        # print(best_predict)
    if args.confusion_matrix:
        print('plotting confusion matrix')
        plotconfusion(best_label,best_predict,log_dir)

       
        

    # x1 = range(1,args.epoch+1)
    # y1 = test_acc_history
    
    # font1 = {'family':'Times New Roman', 'weight':'normal', 'size':12}
    # plt.plot(x1,y1,color='b',label='mAP')
    # # plt.title('Test accuracy vs epoches')
    # plt.legend(loc='lower right', prop=font1, frameon=False)
    # plt.ylabel('Acc')
    # plt.xlabel('Epoch')
    # plt.savefig('/home/Eric/research/TD-PN-DCNN-main/log/'+args.dataset+'/norm_feature_clustering/'+args.data_num+'/td_dcnn_ucm'+args.data_num+'_acc'+args.experiment_time+'.png')
           
       
