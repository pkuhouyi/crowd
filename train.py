
from torch.autograd import Variable
from torchvision import datasets, transforms
import argparse
import json
import time
from utils.utils import AverageMeter
import os
import torch.nn as nn
from models.CSR_net import CSRNet
from torchvision import transforms
from torch.utils.data import DataLoader
from datasets.transforms import *
from datasets.SHHA.SHHA import SHHA
import torch
import random
from utils.utils import save_checkpoint


parser = argparse.ArgumentParser(description='PyTorch CSRNet')
parser.add_argument('--original_lr',type=float, default=1e-9)
parser.add_argument('--lr',type=float, default=1e-7)
parser.add_argument('--batch_size', type=int, default=8)
parser.add_argument('--momentum', type=float, default=0.9)
parser.add_argument('--decay', type=float,default=5*1e-4)
parser.add_argument('--start_epoch', type=int, default=0)
parser.add_argument('--epochs', type=int, default=400)
parser.add_argument('--resume', default=None, type=str,help='path to the pretrained model')
parser.add_argument('--steps', type=int, default=100)
parser.add_argument('--print_freq', type=int, default=1)
parser.add_argument('--seed', type=int, default=100)
parser.add_argument('--train_data_path', default='/home/houyi/datasets/ShanghaiTech/part_B/train_data/', type=str)
parser.add_argument('--test_data_path', default='/home/houyi/datasets/ShanghaiTech/part_B/test_data/', type=str)


args = parser.parse_args()



def get_min_size(batch_imgs):
    '''
    去最小值，不会小于(480,640)
    :param batch_imgs:
    :return:
    '''
    min_ht = 480
    min_wd = 640
    for img in batch_imgs:
        _,ht,wd = img.shape
        if ht<min_ht:
            min_ht = ht
        if wd<min_wd:
            min_wd = wd
    return min_ht,min_wd

def random_crop(dst_size, img, den, dot=None):
    '''
    裁剪到指定的大小，可以确保在一个batch中大小是可以互相匹配的
    :param dst_size:
    :param img:
    :param den:
    :param dot:
    :return:
    '''
    factor=8
    if dot==None:
        _,ts_hd,ts_wd = img.shape

        x1 = random.randint(0, ts_wd - dst_size[1])//factor
        y1 = random.randint(0, ts_hd - dst_size[0])//factor
        x2 = x1 + dst_size[1]
        y2 = y1 + dst_size[0]

        label_x1 = x1//factor
        label_y1 = y1//factor
        label_x2 = x2//factor
        label_y2 = y2//factor

        return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2]
    else:
        _,ts_hd,ts_wd = img.shape

        x1 = random.randint(0, ts_wd - dst_size[1])//factor
        y1 = random.randint(0, ts_hd - dst_size[0])//factor
        x2 = x1 + dst_size[1]
        y2 = y1 + dst_size[0]

        label_x1 = x1//factor
        label_y1 = y1//factor
        label_x2 = x2//factor
        label_y2 = y2//factor

        return img[:,y1:y2,x1:x2], den[label_y1:label_y2,label_x1:label_x2], dot[label_y1:label_y2,label_x1:label_x2]


def collate_func(batch):

    transposed = list(zip(*batch))
    if len(transposed)==2:
        imgs, dens = [transposed[0],transposed[1]]
        min_ht, min_wd = get_min_size(imgs)
        cropped_imgs = []
        cropped_dens = []
        for i_sample in range(len(batch)):
            _img, _den = random_crop([min_ht,min_wd], imgs[i_sample],dens[i_sample])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)

        cropped_imgs = torch.stack(cropped_imgs)
        cropped_dens = torch.stack(cropped_dens)

        return [cropped_imgs,cropped_dens]
    else:
        imgs, dens ,dots= [transposed[0],transposed[1], transposed[2]]
        min_ht, min_wd = get_min_size(imgs)
        cropped_imgs = []
        cropped_dens = []
        cropped_dots = []
        for i_sample in range(len(batch)):
            _img, _den, _dots = random_crop([min_ht,min_wd],imgs[i_sample],dens[i_sample],dots[i_sample])
            cropped_imgs.append(_img)
            cropped_dens.append(_den)
            cropped_dots.append(_dots)

        cropped_imgs = torch.stack(cropped_imgs)
        cropped_dens = torch.stack(cropped_dens)
        cropped_dots = torch.stack(cropped_dots)
        return [cropped_imgs, cropped_dens, cropped_dots]


def train(train_loader, model, criterion, optimizer, epoch):

    losses = AverageMeter()
    batch_time = AverageMeter()
    data_time = AverageMeter()

    print('epoch %d, processed %d samples, lr %.10f' % (epoch, epoch * len(train_loader.dataset), args.lr))

    model.train()
    end = time.time()

    for i,(img, target)in enumerate(train_loader):
        print(i)
        data_time.update(time.time() - end)

        img = img.cuda()
        img = Variable(img)
        output = model(img)


        target = target.type(torch.FloatTensor).unsqueeze(0).cuda()
        target = Variable(target)

        loss = criterion(output, target)
        losses.update(loss.item(), img.size(0))
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        batch_time.update(time.time() - end)
        end = time.time()

        if i % args.print_freq == 0:
            print('Epoch: [{0}][{1}/{2}]\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                .format(
                epoch, i, len(train_loader), batch_time=batch_time,
                data_time=data_time, loss=losses))



def validate(val_loader, model, criterion):
    print ('begin test')

    model.eval()
    mae = 0
    for i,(img, target) in enumerate(val_loader):
        img = img.cuda()
        img = Variable(img)
        output = model(img)
        mae += abs(output.data.sum()-target.sum().type(torch.FloatTensor).cuda())
    mae = mae/len(val_loader)
    print(' * MAE {mae:.3f} '.format(mae=mae))
    return mae

def adjust_learning_rate(optimizer, epoch):
    """Sets the learning rate to the initial LR decayed by 10 every 30 epochs"""
    args.lr = args.original_lr
    args.steps=[-1,1,100,150]
    args.scales = [1,1,1,1]
    for i in range(len(args.steps)):
        scale = args.scales[i] if i < len(args.scales) else 1
        if epoch >= args.steps[i]:
            args.lr = args.lr * scale
            if epoch == args.steps[i]:
                break
        else:
            break
    for param_group in optimizer.param_groups:
        param_group['lr'] = args.lr

def main():
    os.environ['CUDA_VISIBLE_DEVICES'] = '0'
    torch.cuda.manual_seed(args.seed)


    mean_std = ([0.537967503071, 0.460666239262, 0.41356408596],[0.220573320985, 0.218155637383, 0.20540446043])
    log_para = 100.
    factor = 8
    train_main_transform = Compose([RandomHorizontallyFlip()])
    img_transform = transforms.Compose([
        transforms.ToTensor(),
        transforms.Normalize(*mean_std)
    ])
    gt_transform = transforms.Compose([
        GTScaleDown(factor),
        LabelNormalize(log_para)
    ])

    train_set = SHHA(args.train_data_path,main_transform=train_main_transform, img_transform=img_transform, density_transform=gt_transform)
    train_loader = DataLoader(train_set, batch_size=args.batch_size, num_workers=8, collate_fn=collate_func, shuffle=True, drop_last=True)


    val_set = SHHA(args.test_data_path, main_transform=None, img_transform=img_transform, density_transform=gt_transform)
    val_loader = DataLoader(val_set, batch_size=args.batch_size, num_workers=8, shuffle=True, drop_last=False)


    model = CSRNet()
    model = model.cuda()
    criterion = nn.MSELoss(size_average=False).cuda()
    optimizer = torch.optim.SGD(model.parameters(), args.lr,momentum=args.momentum,weight_decay=args.decay)

    best_prec1=0
    if args.resume!=None:
        if os.path.isfile(args.pre):
            print("=> loading checkpoint '{}'".format(args.pre))
            checkpoint = torch.load(args.pre)
            args.start_epoch = checkpoint['epoch']
            best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            optimizer.load_state_dict(checkpoint['optimizer'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.pre, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.pre))

    for epoch in range(args.start_epoch, args.epochs):

        # adjust_learning_rate(optimizer, epoch)
        train(train_loader, model, criterion, optimizer, epoch)
        # prec1 = validate(val_loader, model, criterion)
        #
        # is_best = prec1 < best_prec1
        # best_prec1 = min(prec1, best_prec1)
        # print(' * best MAE {mae:.3f} '.format(mae=best_prec1))
        #
        # save_checkpoint({
        #     'epoch': epoch + 1,
        #     'arch': args.pre,
        #     'state_dict': model.state_dict(),
        #     'best_prec1': best_prec1,
        #     'optimizer' : optimizer.state_dict(),
        # }, is_best,args.task)


if __name__ == '__main__':
    main()