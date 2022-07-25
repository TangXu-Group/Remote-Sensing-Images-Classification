import argparse
import matplotlib.pyplot as plt
import os
import torch
import torch.nn as nn
from torchvision import transforms
from torch.utils.data import DataLoader
from tqdm import tqdm
from timm.scheduler import create_scheduler

from dataset.custom_data import MyCustomDataset
from models.resnet34_mvt import ResNet34Mvt


os.environ['CUDA_VISIBLE_DEVICES'] = '1'
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def get_args_parser():
    root_path = '/home/amax/lmt/pytorch_project/EMTCAL/'
    parser = argparse.ArgumentParser('ResNet training and evaluation script', add_help=False)
    parser.add_argument('--batch_size', default=32, type=int)
    parser.add_argument('--epochs', default=100, type=int)

    # Model parameters
    parser.add_argument('--model', default='resnet34_mvt', type=str, help='Name of model to train')
    parser.add_argument('--image_size', default=224, type=int, help='images input size')
    parser.add_argument('--drop', type=float, default=0.0, help='Dropout rate (default: 0.0)')

    # Optimizer parameters
    parser.add_argument('--opt', default='adamw', type=str, metavar='OPTIMIZER',
                        help='Optimizer (default: "adamw"')
    parser.add_argument('--opt_eps', default=1e-8, type=float, metavar='EPSILON',
                        help='Optimizer Epsilon (default: 1e-8)')

    # Learning rate schedule parameters
    parser.add_argument('--sched', default='cosine', type=str, metavar='SCHEDULER',
                        help='LR scheduler (default: "cosine"')
    parser.add_argument('--lr', type=float, default=0.00003, metavar='LR',
                        help='learning rate (default: 0.00003)')
    parser.add_argument('--warmup_lr', type=float, default=1e-6, metavar='LR',
                        help='warmup learning rate (default: 1e-6)')
    parser.add_argument('--min_lr', type=float, default=1e-5, metavar='LR',
                        help='lower lr bound for cyclic schedulers that hit 0 (1e-5)')
    parser.add_argument('--warmup_epochs', type=int, default=5, metavar='N',
                        help='epochs to warmup LR, if scheduler supports')
    parser.add_argument('--cooldown_epochs', type=int, default=0, metavar='N',
                        help='epochs to cooldown LR at min_lr, after cyclic schedule ends')
    parser.add_argument('--decay_rate', '--dr', type=float, default=0.1, metavar='RATE',
                        help='LR decay rate (default: 0.1)')

    # Dataset parameters
    parser.add_argument("--dataset", default="UC_Merced",
                        help="Which dataset.")
    parser.add_argument("--dataset_path", default="UC_Merced82",
                        help="dataset path.")
    parser.add_argument('--data_dir', default=root_path + 'dataset', type=str)
    parser.add_argument('--log_dir', default=root_path + 'log', type=str)
    parser.add_argument("--parameters", default="parameters", type=str,
                        help="The output directory where checkpoints will be written.")
    parser.add_argument('--seed', default=42, type=int)
    parser.add_argument('--num_workers', default=4, type=int)

    args = parser.parse_args()
    label_dims = {'UC_Merced': 21, 'NWPU': 45, 'AID': 30, 'RSSDIVCS': 70}
    args.label_dim = label_dims[args.dataset]
    args.img_tr = os.path.join(args.data_dir, args.dataset_path, 'train.txt')
    args.img_te = os.path.join(args.data_dir, args.dataset_path, 'test.txt')
    return args

def main():
    args = get_args_parser()
    torch.manual_seed(42)
    train_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.RandomHorizontalFlip(),
        transforms.RandomVerticalFlip(),
        transforms.RandomRotation(30),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225])
    ])
    test_transform = transforms.Compose([
        transforms.Resize((args.image_size, args.image_size)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406],
                             std=[0.229, 0.224, 0.225])
    ])

    # loading the training and test set
    train_data = MyCustomDataset(args.img_tr, transform=train_transform)
    test_data = MyCustomDataset(args.img_te, transform=test_transform)
    print('\ntest_Data', len(test_data))
    print(args.label_dim)

    train_loader = DataLoader(train_data,
                              batch_size=args.batch_size,
                              shuffle=True,
                              num_workers=4)
    test_loader = DataLoader(test_data,
                             batch_size=args.batch_size,
                             shuffle=False,
                             num_workers=4)


    model = ResNet34Mvt(num_class=args.label_dim)
    model.to(device)

    num_params = count_parameters(model)
    print("Training parameters:%s",args)
    print("Total Parameter: \t%2.1fM" % num_params)

    optimizer = torch.optim.Adam(model.parameters(),lr=args.lr,eps=args.opt_eps)
    scheduler, _ = create_scheduler(args, optimizer)

    criteon = nn.CrossEntropyLoss()

    best_acc, best_epoch = 0.0, 0
    loss_show = []
    acc_show = []

    for epoch in range(args.epochs):
        model.train()
        description = str(epoch) + "/" + str(args.epochs)
        with tqdm(train_loader, desc=description) as iterator:
            for img, label in iterator:
                img, label = img.to(device), label.to(device)
                logits = model(img)

                loss = criteon(logits, label)
                optimizer.zero_grad()

                loss.backward()
                optimizer.step()

                predict_label = torch.argmax(logits, dim=1)
                train_acc = (predict_label == label).sum().float().item() / len(label)
                information = "ta:{:.4f},loss:{:.4f},lr:{:.6f}".format(train_acc, loss.item(),
                                                                       optimizer.state_dict()["param_groups"][0]["lr"])
                iterator.set_postfix_str(information)
                loss_show.append(loss)
        scheduler.step(epoch)

        if epoch % 1 == 0:
            test_acc = evalute(model, test_loader)
            if test_acc > best_acc:
                best_acc = test_acc
                best_epoch = epoch
                torch.save(model.state_dict(), os.path.join(args.parameters, "%s_checkpoint.pth" % args.dataset))

            print('Epoch:', epoch, '  test_acc:', test_acc, 'loss:', loss.item())
            acc_show.append(test_acc)

            f = open(os.path.join(args.log_dir, args.model, 'ro30_'+args.dataset_path+'.log'), 'a')
            f.write('Epoch:' + str(epoch) + '   acc:' + str(test_acc) + '   Bestacc:' + str(best_acc) + '\n')
            f.close()

        # scheduler.step()
    print('best acc:', best_acc, 'best epoch:', best_epoch)
    show_loss(loss_show)
    show_acc(acc_show)

# 测试OA准确率
def evalute(model, loader):
    model.eval()
    correct = 0.0
    total = len(loader.dataset)
    for img, label in loader:
        img, label = img.to(device), label.to(device)
        with torch.no_grad():
            out = model(img)
            pred = out.argmax(dim=1)
        correct += torch.eq(pred, label).sum().float().item()
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
    return params / 1000000

if __name__ == '__main__':
    main()




