import argparse
import torch
import os
from torch.utils.data import DataLoader
from UCM import UCMercedLand

from sampler import CategoriesSampler
from convnet import Convnet,S_Classifier,R_Classifier
from utils import Timer, Averager, count_acc, euclidean_metric
# from Batch_Folding import BatchFold
# from ProtoReshape import AdaptiveAddition
import torchvision.transforms.functional as TF

from GAM import GAM_Attention
import torch.nn.functional as F
device_ids = [0,1]
torch.multiprocessing.set_sharing_strategy('file_system')
torch.cuda.manual_seed(1)
# device = torch.device("cuda:3" if torch.cuda.is_available() else "cpu")
# print(device)

def log(path,filename,data):
    with open(os.path.join(path,filename),'w') as f:
        for item in data:
            f.write("%.4f\n"%item)
        
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

if __name__=='__main__':
    parser = argparse.ArgumentParser()
    parser.add_argument('--episode', type=int, default=500)
    parser.add_argument('--shot', type=int, default=1)
    parser.add_argument('--query', type=int, default=15)
    parser.add_argument('--test_way', type=int, default=5)
    parser.add_argument('--numworkers', type=int, default=8)
    parser.add_argument('--dataset', type=str, default='NWPU')
    parser.add_argument('--experiment_time',type=str, default='15')
    parser.add_argument('--backbone',type=str, default='Convnet')
    parser.add_argument('--alpha',type=float, default=1.0)
    parser.add_argument('--beta',type=float, default=1.0)
    parser.add_argument('--ld',type=float, default=0.1)
    parser.add_argument('--embed_dim',type=float, default=64)
    parser.add_argument('--trans_num',type=int, default=4)
    parser.add_argument('--scale1',type=int, default=224)
    parser.add_argument('--scale2',type=int, default=234)
    parser.add_argument('--scale3',type=int, default=244)
    parser.add_argument('--hppair',type=int, default=1)
    parser.add_argument('--metric', type=str, default='euclidean', choices=['euclidean','cosine_similarity'])
    args = parser.parse_args()
    print(vars(args))

    checkpoint_path={
        'UC_Merced':"/home/amax/Eric/ISPRS/checkpoint/hp/UC_Merced/",
        'AID':"/home/amax/Eric/ISPRS/checkpoint/pre_hp/AID/",
        'NWPU':"/home/Eric/research/MPCL/best/NWPU/"
    }

    # checkpoint_path = "/home/Eric/research/MPCL/best/AID/"

    log_path={
        'UC_Merced':"/home/Eric/research/MPCL/log_cross/UC_Merced/",
        'AID':"/home/Eric/research/MPCL/log_cross/AID/",
        'NWPU':"/home/Eric/research/MPCL/log_cross/NWPU/"
    }

    csv_path = {
        'UC_Merced':"/home/Eric/research/MPCL/data_csv/UC_Merced/",
        'AID':"/home/Eric/research/MPCL/data_csv/AID/",
        'NWPU':"/home/Eric/research/MPCL/data_csv/NWPU/"
    }

    data_path={
        'UC_Merced':"/home/Eric/research/TD-PN-DCNN-main/UC_Merced",
        'AID':"/home/Eric/research/CPT/AID",
        'NWPU':"/home/Eric/research/TD-PN-DCNN-main/NWPU"
        }

    
    data_dir = data_path[args.dataset]
    log_dir = log_path[args.dataset]+str(args.test_way)+'way_'+str(args.shot)+'shot/'
    csv_dir = csv_path[args.dataset]

    txt_name='test_acc_history'+args.experiment_time+'.txt'

    testset = UCMercedLand(data_dir, csv_dir, 'test')
    test_sampler = CategoriesSampler(testset.label, args.episode,
                                      args.test_way, args.shot+args.query)
    test_loader = DataLoader(dataset=testset, batch_sampler=test_sampler,
                              num_workers = args.numworkers)

    

    checkpoint_dir = checkpoint_path[args.dataset]+str(args.test_way)+'_way'+str(args.shot)+'_shot'+'_model'+'.pth'
    checkpoint = torch.load(checkpoint_dir)
    # model = resnet18(include_top=False).to(device)
    model = Convnet()
    model = torch.nn.DataParallel(model, device_ids=device_ids)
    model = model.cuda(device=device_ids[0])
    model.load_state_dict(checkpoint['model'])
    
    scale_classifier = S_Classifier()
    scale_classifier.load_state_dict(checkpoint['scale_classifier'])
    scale_classifier = torch.nn.DataParallel(scale_classifier, device_ids=device_ids)
    scale_classifier = scale_classifier.cuda(device=device_ids[0])
    rot_classifier = R_Classifier()
    rot_classifier.load_state_dict(checkpoint['rot_classifier'])
    rot_classifier = torch.nn.DataParallel(rot_classifier, device_ids=device_ids)
    rot_classifier = rot_classifier.cuda(device=device_ids[0])
   

    model.eval()
    scale_classifier.eval()
    rot_classifier.eval()
    

    
    ave_acc = Averager()
    best_acc = 0.0
    test_acc_history = []

    # 记录实验设置
    with open(os.path.join(log_dir, 'setting'+args.experiment_time+'.txt'), 'w') as f:
            for k,v in vars(args).items():
                f.write('{}:{}\n'.format(k,v))
            f.close()

    with torch.no_grad():
        for i, batch in enumerate(test_loader, 1):
            scale_84, _ = [_.cuda() for _ in batch]
            p = args.shot * args.test_way
            
            scale_104 = TF.resize(scale_84, (args.scale1,args.scale1))
            scale_124 = TF.resize(scale_84, (args.scale2,args.scale2))
            scale_144 = TF.resize(scale_84, (args.scale3,args.scale3))
        
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
            proto_84 = shot_84.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_84 = euclidean_metric(query_84, proto_84)

            shot_104, query_104 = scale_104_fs[:p], scale_104_fs[p:]
            proto_104 = shot_104.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_104 = euclidean_metric(query_104, proto_104)

            shot_124, query_124 = scale_124_fs[:p], scale_124_fs[p:]
            proto_124 = shot_124.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_124 = euclidean_metric(query_124, proto_124)

            shot_144, query_144 = scale_144_fs[:p], scale_144_fs[p:]
            proto_144 = shot_144.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_144 = euclidean_metric(query_144, proto_144)

            logits_dict = [logits_84, logits_104, logits_124, logits_144]
            scale_fs_dict = [scale_84_fs, scale_104_fs, scale_124_fs, scale_144_fs]
            # logits = torch.cat(logits, 0)
            scale_fs = torch.cat(scale_fs_dict, 0)

            ## 1nd: scale label
            scale_label = torch.arange(args.trans_num, dtype=torch.int8).view(-1, 1).repeat(1, args.query*args.test_way+args.shot*args.test_way).type(torch.LongTensor)
            scale_label = scale_label.view(-1).cuda()
            ## 2nd:  fsl label
            fsl_label = torch.arange(args.test_way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor).cuda()

            # scale loss
            
            scale_pred = scale_classifier(scale_fs)
            # scale_loss = criterion(scale_pred, scale_label)

            # MI mutual information:  KD loss
            # raw_logits = sum(logits_dict) / len(logits_dict)
            # raw_logits = F.log_softmax(raw_logits, -1)
            # MI_losses = [F.kl_div(raw_logits, F.softmax(logits, -1), size_average=True) for logits in logits_dict]
            # scale_MI_loss = sum(MI_losses) / len(MI_losses)

            # fsl loss for all the tasks copy
            # scale_fsl_losses = [F.cross_entropy(logits, fsl_label) for logits in logits_dict]
            # scale_fsl_loss = sum(scale_fsl_losses) / len(scale_fsl_losses)
            #fsl_loss = F.cross_entropy(raw_logits, fsl_label)
            
            acc_list = [count_acc(logits, fsl_label) for logits in logits_dict] # for 4 single angles tasks
            final_scale_acc = sum(acc_list)/len(acc_list)

            # rotation self-supervision
            shot_0, query_0 = rot_0_fs[:p], rot_0_fs[p:]
            proto_0 = shot_0.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_0 = euclidean_metric(query_0, proto_0)

            shot_90, query_90 = rot_90_fs[:p], rot_90_fs[p:]
            proto_90 = shot_90.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_90 = euclidean_metric(query_90, proto_90)

            shot_180, query_180 = rot_180_fs[:p], rot_180_fs[p:]
            proto_180 = shot_180.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_180 = euclidean_metric(query_180, proto_180)

            shot_270, query_270 = rot_270_fs[:p], rot_270_fs[p:]
            proto_270 = shot_270.reshape(args.shot, args.test_way, -1).mean(dim=0)
            logits_270 = euclidean_metric(query_270, proto_270)

            logits_dict = [logits_0, logits_90, logits_180, logits_270]
            rot_fs_dict = [rot_0_fs, rot_90_fs, rot_180_fs, rot_270_fs]
            # logits = torch.cat(logits, 0)
            rot_fs = torch.cat(rot_fs_dict, 0)

            ## 1nd: rotation label
            rot_label = torch.arange(args.trans_num, dtype=torch.int8).view(-1, 1).repeat(1, args.query*args.test_way+args.shot*args.test_way).type(torch.LongTensor)
            rot_label = rot_label.view(-1).cuda()
            ## 2nd:  fsl label
            fsl_label = torch.arange(args.test_way, dtype=torch.int8).repeat(args.query).type(torch.LongTensor).cuda()

            # rotation loss
            
            rot_pred = rot_classifier(rot_fs)
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
            acc_list.append(final_acc)
        
            acc = acc_list[0]
            ave_acc.add(acc)
        
            print('batch {}: acc: {:.2f} avg_acc: {:.2f}'.format(i,  acc * 100, ave_acc.item() * 100 ))
            test_acc_history.append(ave_acc.item())
            # log(log_dir,txt_name,test_acc_history)
