# coding=utf-8
from __future__ import absolute_import, division, print_function
import logging
import argparse
import os
import numpy as np
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
import matplotlib.pyplot as plt
from utils.scheduler import WarmupLinearSchedule, WarmupCosineSchedule
from dataset.custom_data import MyCustomDataset
from models.modeling import CONFIGS
from models.conv_combine_model import CombineVisionTransformerc
from loss.contrastive_loss import con_loss

print(torch.__version__)

os.environ['CUDA_VISIBLE_DEVICES'] = '3'
logger = logging.getLogger(__name__)
device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
torch.manual_seed(42)
# torch.cuda.set_device(1)


def train(args):
    # train_loader, test_loader = get_loader(args)
    train_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])
    test_transform = transforms.Compose([
        transforms.Resize((384, 384)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5]),
    ])

    # loading the training and test set
    train_data = MyCustomDataset(root_path=args.img_tr, slic_path=args.img_slic_tr,
                                 transform=train_transform)
    test_data = MyCustomDataset(root_path=args.img_te, slic_path=args.img_slic_te,
                                transform=test_transform)
    print('\ntest_Data',len(test_data))
    print('\nnum_class:',args.label_dim)

    train_loader = DataLoader(train_data,
                              batch_size=args.train_batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_data,
                             batch_size=args.eval_batch_size,
                             shuffle=False,
                             num_workers=4)

    # loading the network
    config = CONFIGS[args.model_type]
    model = CombineVisionTransformerc(config, args.img_size, zero_head=True,
                                     num_classes=args.label_dim, pretrained_dir=args.pretrained_dir)
    model.to(device)
    num_params = count_parameters(model)

    logger.info("{}".format(config))
    logger.info("Training parameters %s", args)
    logger.info("Total Parameter: \t%2.1fM" % num_params)

    # Prepare optimizer and scheduler
    optimizer = torch.optim.SGD(model.parameters(),
                                lr=args.learning_rate,
                                momentum=0.9,
                                weight_decay=args.weight_decay)

    t_total = args.num_steps

    if args.decay_type == "cosine":
        scheduler = WarmupCosineSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)
    else:
        scheduler = WarmupLinearSchedule(optimizer, warmup_steps=args.warmup_steps, t_total=t_total)

    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0.0, 0
    loss_show = []
    acc_show = []

    for epoch in range(args.epoch):
        model.train()
        description = str(epoch) + "/" + str(args.epoch)
        with tqdm(train_loader, desc=description) as iterator:
            for step, (img, img_slic, label) in enumerate(iterator):
                img, img_slic, label = img.to(device), img_slic.to(device), label.to(device)
                logits, feature_yuan, feature_slic=model(img,img_slic)
                
#                 loss = criteon(logits, label)
                
                #new_dual_feature
                ce_loss = criteon(logits, label)
                contrast0_loss = con_loss(feature_yuan, label)
                contrast1_loss = con_loss(feature_slic, label)
                loss = ce_loss + (contrast0_loss + contrast1_loss) * 0.5
                
#                 print('ce_loss:',ce_loss)
                

                scheduler.step()

                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                
                predict_label = torch.argmax(logits,dim=1)
                train_acc = (predict_label==label).sum().float().item() / len(label)
                information = "ta:{:.4f},loss:{:.4f},lr:{:.6f}".format(train_acc,loss.item(),optimizer.state_dict()["param_groups"][0]["lr"])
                iterator.set_postfix_str(information)
                loss_show.append(loss)
            
        if epoch % 1 == 0:
            test_acc = evalute(model,test_loader)
            
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(model.state_dict(),os.path.join(args.parameters, "%s_checkpoint.pth" % args.name))

            print('Epoch:', epoch, '  test_acc:', test_acc, 'loss:', loss.item())
            acc_show.append(test_acc)

            f = open(os.path.join(args.log_dir,'save_model_m_0.5_new_dccl_'+args.dataset+'_slic.log'), 'a')
            f.write('Epoch:' + str(epoch) + '   acc:' + str(test_acc) + '   Bestacc:' + str(best_acc) + '\n')
            f.close()
            
    print('best acc:', best_acc, 'best epoch:', best_epoch)
    show_loss(loss_show)
    show_acc(acc_show)

#测试OA准确率
def evalute(model, loader):
    model.eval()
    correct = 0.0
    total = len(loader.dataset)
    for img,img_slic,label in loader:
        img, img_slic, label = img.to(device), img_slic.to(device),label.to(device)
        with torch.no_grad():
            out = model(img,img_slic)[0]
            pred = out.argmax(dim=1)
        correct += torch.eq(pred,label).sum().float().item()

    return correct / total

#show loss
def show_loss(loss):
    x = range(len(loss))
    y = loss
    plt.plot(x, y, 'o-')
    plt.ylabel('test loss')
    plt.savefig(os.path.join('./log/', 'loss.jpg'))
    plt.close()

#show test_acc
def show_acc(acc):
    x = range(len(acc))
    y = acc
    plt.plot(x, y, 'o-')
    plt.ylabel('test acc')
    plt.savefig(os.path.join('./log/', 'acc.jpg'))
    plt.close()

# num of model's parameters
def count_parameters(model):
    params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    return params/1000000

def main():
    root_path = '/home/admin1/lmt/learning_project/transformer_ViT/'

    parser = argparse.ArgumentParser()
    # Required parameters
    parser.add_argument("--name", default='UC_Merced82_best',
                        help="Name of this run. Used for monitoring.")
    parser.add_argument("--dataset", default="NWPU19_1",
                        help="data path.")
    parser.add_argument("--dataset_slic", default="NWPU19_1_slic",
                        help="slic path.")
    parser.add_argument('--data_dir', default=root_path + 'dataset', type=str)
    parser.add_argument("--model_type", choices=["ViT-B_16", "ViT-B_32", "ViT-L_16",
                                                 "ViT-L_32", "ViT-H_14", "R50-ViT-B_16"],
                        default="ViT-B_32",
                        help="Which variant to use.")
    parser.add_argument("--pretrained_dir", type=str, default="checkpoint/ViT-B_32.npz",
                        help="Where to search for pretrained ViT models.")
    parser.add_argument("--parameters", default="parameter", type=str,
                        help="The output directory where checkpoints will be written.")

    parser.add_argument("--img_size", default=384, type=int,
                        help="Resolution size")
    parser.add_argument("--train_batch_size", default=32, type=int,
                        help="Total batch size for training.")
    parser.add_argument("--eval_batch_size", default=64, type=int,
                        help="Total batch size for eval.")

    parser.add_argument("--learning_rate", default=3e-2, type=float,
                        help="The initial learning rate for SGD.")
    parser.add_argument("--weight_decay", default=0, type=float,
                        help="Weight deay if we apply some.")
    parser.add_argument("--num_steps", default=20000, type=int,
                        help="Total number of training epochs to perform.")
    parser.add_argument('--epoch', default=300, type=int)
    parser.add_argument("--decay_type", choices=["cosine", "linear"], default="cosine",
                        help="How to decay the learning rate.")
    parser.add_argument("--warmup_steps", default=500, type=int,
                        help="Step of training to perform learning rate warmup for.")
    parser.add_argument("--show_log", type=int, default=-1,
                        help="local_rank for distributed training on gpus")
    parser.add_argument('--seed', type=int, default=42,
                        help="random seed for initialization")
    parser.add_argument('--log_dir', default=root_path + 'log/new_log', type=str)
    args = parser.parse_args()

    label_dict = {'UC_Merced': 21, 'NWPU': 45, 'AID': 30, 'UC_Merced55_1': 21,'UC_Merced55_2': 21,'UC_Merced55_3': 21,'UC_Merced55_4': 21,'UC_Merced55_5': 21,'UC_Merced82_n1': 21, 'AID55_2': 30,'AID28_4': 30,'AID55_2': 30,'AID55_5': 30,'AID55seed1': 30, 'NWPU19_e1': 45,'NWPU19_7': 45, 'NWPU28_5': 45, 'NWPU28_6': 45,'NWPU28_7': 45,'RSSDIVCS19_1': 70,}
    args.label_dim = label_dict[args.dataset]
    args.img_tr = os.path.join(args.data_dir, args.dataset, 'train.txt')
    args.img_te = os.path.join(args.data_dir, args.dataset, 'test.txt')
    args.img_slic_tr = os.path.join(args.data_dir, args.dataset_slic, 'train.txt')
    args.img_slic_te = os.path.join(args.data_dir, args.dataset_slic, 'test.txt')

    # Setup logging
    logging.basicConfig(format='%(asctime)s - %(levelname)s - %(name)s - %(message)s',
                        datefmt='%m/%d/%Y %H:%M:%S',
                        level=logging.INFO if args.show_log in [-1, 0] else logging.WARN)

    # Training
    train(args)


if __name__ == "__main__":
    main()





