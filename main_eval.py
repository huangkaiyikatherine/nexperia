import argparse
import os
import shutil
import time
import copy
import PIL.Image as Image
import numpy as np
import pandas as pd

import torch
import torch.nn as nn
from torch.nn import functional as F
import torch.nn.parallel
import torch.backends.cudnn as cudnn

from datasets import get_loader
from losses import get_loss
from models import get_model
from utils import *


parser = argparse.ArgumentParser(description='Self-Adaptive Trainingn')
# network
parser.add_argument('--arch', '-a', metavar='ARCH', default='resnet34',
                    help='model architecture')
parser.add_argument('--weight-decay', '--wd', default=5e-4, type=float,
                    metavar='W', help='weight decay (default: 5e-4)')
parser.add_argument('--base-width', default=64, type=int,
                    help='base width of resnets or hidden dim of fc nets')
# training setting
parser.add_argument('--data-root', help='The directory of data',
                    default='~/datasets/CIFAR10', type=str)
parser.add_argument('--data-root-1', help='The directory of data',
                    default='~/datasets/CIFAR10', type=str)
parser.add_argument('--data-root-2', help='The directory of data',
                    default='~/datasets/CIFAR10', type=str)
parser.add_argument('--dataset', help='dataset used to training',
                    default='cifar10', type=str)
parser.add_argument('--train-sets', help='subsets (train/trainval) that used to training',
                    default='train', type=str)
parser.add_argument('--val-sets', type=str, nargs='+', default=['noisy_val'],
                    help='subsets (clean_train/noisy_train/clean_val/noisy_val) that used to validation')
parser.add_argument('--train-set', help='subset (train/trainval) that used to train',
                    default='train', type=str)
parser.add_argument('--val-set', help='subsets (val) that used to validate',
                    default='val', type=str)
parser.add_argument('--test-set', help='subsets (test) that used to test',
                    default='test', type=str)
parser.add_argument('--crop', help='crop style',
                    default='center', type=str)
parser.add_argument('--epochs', default=200, type=int, metavar='N',
                    help='number of total epochs to run')
parser.add_argument('--start-epoch', default=0, type=int, metavar='N',
                    help='manual epoch number (useful on restarts)')
parser.add_argument('--resume', default='', type=str, metavar='PATH',
                    help='path to latest checkpoint (default: none)')
parser.add_argument('-b', '--batch-size', default=128, type=int,
                    metavar='N', help='mini-batch size (default: 128)')
parser.add_argument('--optimizer', default='sgd', type=str,
                    help='optimizer for training')
parser.add_argument('--lr', '--learning-rate', default=0.1, type=float,
                    metavar='LR', help='initial learning rate')
parser.add_argument('--lr-schedule', default='step', type=str,
                    help='LR decay schedule')
parser.add_argument('--lr-milestones', type=int, nargs='+', default=[40, 80],
                    help='LR decay milestones for step schedule.')
parser.add_argument('--lr-gamma', default=0.1, type=float,
                    help='LR decay gamma')
parser.add_argument('--momentum', default=0.9, type=float, metavar='M',
                    help='momentum')
# noisy setting
parser.add_argument('--noise-rate', default=0., type=float,
                    help='Label noise rate')
parser.add_argument('--noise-type', default=None, type=str,
                    help='Noise type, could be one of (corrupted_label, Gaussian, random_pixels, shuffled_pixels)')
parser.add_argument('--noise-info', default=None, type=str, 
                    help='directory of pre-configured noise pattern.')
parser.add_argument('--use-refined-label', action='store_true', help='whether or not use refined label by self-adaptive training')
parser.add_argument('--turn-off-aug', action='store_true', help='whether or not use data augmentation')
# loss function
parser.add_argument('--loss', default='ce', help='loss function')
parser.add_argument('--fl-lambda', default=0.5, type=float,
                    help='lambda of sat focal loss')
parser.add_argument('--fl-alpha', default=1, type=float,
                    help='alpha of focal loss')
parser.add_argument('--fl-gamma', default=2, type=float,
                    help='gamma of focal loss')
parser.add_argument('--sat-alpha', default=0.9, type=float,
                    help='momentum term of self-adaptive training')
parser.add_argument('--sat-es', default=0, type=int,
                    help='start epoch of self-adaptive training (default 0)')
parser.add_argument('--sat-es1', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 1')
parser.add_argument('--sat-es2', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 2')
parser.add_argument('--sat-es3', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 3')
parser.add_argument('--sat-es4', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 4')
parser.add_argument('--sat-es5', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 5')
parser.add_argument('--sat-es6', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 6')
parser.add_argument('--sat-es7', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 7')
parser.add_argument('--sat-es8', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 8')
parser.add_argument('--sat-es9', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 9')
parser.add_argument('--sat-es10', default=None, type=int,
                    help='start epoch of self-adaptive training (default 0) for class 10')
# misc
parser.add_argument('-s', '--seed', default=None, type=int,
                    help='number of data loading workers (default: None)')
parser.add_argument('-j', '--workers', default=4, type=int, metavar='N',
                    help='number of data loading workers (default: 4)')
parser.add_argument('--print-freq', '-p', default=50, type=int,
                    metavar='N', help='print frequency (default: 50)')
parser.add_argument('-e', '--evaluate', help='evaluate mode',
                    default=False, type=bool)
parser.add_argument('--save-dir', dest='save_dir',
                    help='The directory used to save the trained models',
                    default='save_temp', type=str)
parser.add_argument('--checkpoint', dest='checkpoint',
                    help='The directory used to load the model',
                    default='load_model', type=str)
parser.add_argument('--load-model', dest='load_model',
                    help='which model to load',
                    default='save_temp', type=str)
parser.add_argument('--save-freq', default=0, type=int,
                    help='print frequency (default: 0, i.e., only best and latest checkpoints are saved)')
parser.add_argument('--mod', default=None, type=str,
                    help='SAT modification')
parser.add_argument('--pretrained', help='whether to use a pretrained model',
                    default=False, type=bool)
parser.add_argument('--feature-extraction', help='whether to train only the last layer',
                    default=False, type=bool)
parser.add_argument('--el1', default=None, type=int,
                    help='early learning for class 1')
parser.add_argument('--el2', default=None, type=int,
                    help='early learning for class 2')
parser.add_argument('--el3', default=None, type=int,
                    help='early learning for class 3')
parser.add_argument('--el4', default=None, type=int,
                    help='early learning for class 4')
parser.add_argument('--el5', default=None, type=int,
                    help='early learning for class 5')
parser.add_argument('--el6', default=None, type=int,
                    help='early learning for class 6')
parser.add_argument('--el7', default=None, type=int,
                    help='early learning for class 7')
parser.add_argument('--el8', default=None, type=int,
                    help='early learning for class 8')
parser.add_argument('--el9', default=None, type=int,
                    help='early learning for class 9')
parser.add_argument('--el10', default=None, type=int,
                    help='early learning for class 10')
parser.add_argument('--ce-momentum', default=0, type=float,
                    help='momentum term of binary weighted CE')
args = parser.parse_args()


best_prec1 = 0
best_auc_1 = 0
best_auc_2 = 0
best_auc_3 = 0
best_model_1 = None
best_model_2 = None
best_model_3 = None
best_epoch_1 = 0
best_epoch_2 = 0
best_epoch_3 = 0
if args.seed is None:
    import random
    args.seed = random.randint(1, 10000)


def main():
    
    print('*'*40)
    print(args.checkpoint)
    print(args.load_model)
    print(args.val_set)
    print('-'*40)
    
    ## dynamically adjust hyper-parameters for ResNets according to base_width
    if args.base_width != 64 and 'sat' in args.loss:
        factor = 64. / args.base_width
        args.sat_alpha = args.sat_alpha**(1. / factor)
        args.sat_es = int(args.sat_es * factor)
        print("Adaptive parameters adjustment: alpha = {:.3f}, Es = {:d}".format(args.sat_alpha, args.sat_es))

    global best_prec1
    global best_auc_1, best_auc_2, best_auc_3, best_model_1, best_model_2, best_model_3, best_epoch_1, best_epoch_2, best_epoch_3

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'train'))
        os.makedirs(os.path.join(args.save_dir, 'val'))
        os.makedirs(os.path.join(args.save_dir, 'test'))

    # prepare dataset
    val_loader, num_classes, val_targets, pass_idx = get_loader(args)
    
    model = get_model(args, num_classes, base_width=args.base_width)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()
    checkpoint = torch.load(args.checkpoint)
    model.load_state_dict(checkpoint[args.load_model])

    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    criterion = get_loss(args, labels=val_targets, num_classes=num_classes,
                         train_len=0, val_len=len(val_targets), test_len=0,
                         pass_idx=pass_idx)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    train_timeline = Timeline()
    val_timeline = Timeline()
    test_timeline = Timeline()
    
    validate(val_loader, model, 0, val_timeline, args.dataset, criterion=criterion, crop=args.crop, last=True)
    return


def validate(val_loader, model, epoch, timeline, dataset, state=None, criterion=None, last=False, crop='center'):
    """
    Run evaluation
    """
    batch_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    binary_losses = AverageMeter()
    binary_acc = AverageMeter()
    margin_error = AverageMeter()
    binary_margin_error = AverageMeter()
    
    auc_meter = AUCMeter()

    # switch to evaluate mode
    model.eval()

    end = time.time()    
    for i, (input, target, index, image_id) in enumerate(val_loader):
        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        with torch.no_grad():
            # compute output
            if crop!='center':
                bs, ncrops, c, h, w = input.size()
                output = model(input.view(-1, c, h, w)).view(bs, ncrops, -1).mean(1)
            else:
                output = model(input)

            probs = F.softmax(output, dim=1)
            if state==None or criterion==None:
                loss = F.cross_entropy(output, target)
                loss_bi = F.binary_cross_entropy(1-probs[:,criterion.pass_idx], (target!=criterion.pass_idx).float())
                me = torch.mean(probs[np.arange(len(target)),target]
                                - torch.max(
                                    probs[
                                        torch.arange(probs.size(1)).reshape(1,-1).repeat(len(target),1)
                                        !=target.reshape(-1,1).repeat(1,probs.size(1)).cpu()]
                                    .view(len(target), -1), 1)[0])
                margin_error_bi = torch.mean((probs[:,criterion.pass_idx] * 2 - 1) * torch.sign((target==criterion.pass_idx).int() - 0.5))
            elif state=='val':
                loss, loss_bi, me, margin_error_bi = criterion(output, target, index, epoch, 'val')
            elif state=='test':
                loss, loss_bi, me, margin_error_bi = criterion(output, target, index, epoch, 'test')
            else:
                raise KeyError("State {} is not supported.".format(state))

            output = output.float()
            loss = loss.float()
            loss_bi = loss_bi.float()

            # measure accuracy and record loss
            prec1 = accuracy(output.data, target)[0]

            acc_bi = torch.sum((probs[:,criterion.pass_idx]<=0.5)==(target!=criterion.pass_idx)) * 100. / input.size(0)

            losses.update(loss.item(), input.size(0))
            top1.update(prec1.item(), input.size(0))
            binary_losses.update(loss_bi.item(), input.size(0))
            binary_acc.update(acc_bi.item(), input.size(0))
            margin_error.update(me.item(), input.size(0))
            binary_margin_error.update(margin_error_bi.item(), input.size(0))

            # measure auc
            auc_meter.update(target.detach(), probs.detach())
            
    # measure elapsed time
    batch_time.update(time.time() - end)
    end = time.time()
                    
    if state is not None:
        acc_class = torch.zeros(1, criterion.num_classes)
        loss_class = torch.zeros(1, criterion.num_classes)
        acc_bi_class = torch.zeros(1, criterion.num_classes)
        loss_bi_class = torch.zeros(1, criterion.num_classes)
        me_class = torch.zeros(1, criterion.num_classes)
        me_bi_class = torch.zeros(1, criterion.num_classes)

        if hasattr(criterion, 'outputs'):
            outputs = criterion.outputs
        else:
            outputs = criterion.soft_labels
            
        if state=='val':
            start = criterion.len_train
            end = criterion.len_train + criterion.len_val
        elif state=='test':
            start = criterion.len_train + criterion.len_val
            end = len(criterion.labels)
            
        for i in range(criterion.num_classes):
            acc_class[0, i] = torch.sum(torch.argmax(outputs[start:end][
                criterion.labels[start:end]==i,:], dim=1)==i).item() / np.count_nonzero(
                criterion.labels[start:end]==i)
            loss_class[0, i] = -torch.mean(torch.log(outputs[start:end][
                criterion.labels[start:end]==i,i])).item()
            outputs_class = outputs[start:end][criterion.labels[start:end]==i]
            me_class[0, i] = torch.mean(outputs_class[:,i] - torch.max(outputs_class[:,np.arange(criterion.num_classes)!=i], dim=1)[0]).item()
            me_bi_class[0, i] = torch.mean((outputs_class[:,criterion.pass_idx] * 2 - 1) * ((i==criterion.pass_idx) - 0.5)) * 2.
            if i!=criterion.pass_idx:
                acc_bi_class[0, i] = torch.sum(outputs[start:end][
                    criterion.labels[start:end]==i,criterion.pass_idx]<=0.5).item() / np.count_nonzero(
                    criterion.labels[start:end]==i)
                loss_bi_class[0, i] = -torch.mean(torch.log(1-outputs[start:end][
                criterion.labels[start:end]==i,criterion.pass_idx])).item()
            else:
                acc_bi_class[0, i] = torch.sum(outputs[start:end][
                    criterion.labels[start:end]==i,criterion.pass_idx]>0.5).item() / np.count_nonzero(
                    criterion.labels[start:end]==i)
                loss_bi_class[0, i] = -torch.mean(torch.log(outputs[start:end][
                criterion.labels[start:end]==i,criterion.pass_idx])).item()
                
    auc, fpr_98, fpr_991, fpr_993, fpr_995, fpr_997, fpr_999, fpr_1 = auc_meter.calculate(criterion.pass_idx)
    print('Epoch: [{}]\t'
          'Loss {:.4f}\t'
          'Prec@1 {:.2f}%\t'
          'Binary Loss {:.4f}\t'
          'Binary Accuracy {:.2f}%\t'
          'Margin Error {:.4f}\t'
          'Binary Margin Error {:.4f}\t'
          'AUC: {:.2f}%\t'
          'FPR(=98%, 99.1%, 99.3%, 99.5%, 99.7%, 100%): {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%'
          .format(epoch+1, losses.avg, top1.avg, binary_losses.avg, binary_acc.avg, margin_error.avg, binary_margin_error.avg,
                  auc*100., fpr_98*100., fpr_991*100., fpr_993*100., fpr_995*100., fpr_997*100., fpr_1*100.))
    if not last:
        timeline.update(losses.avg, top1.avg, binary_losses.avg, binary_acc.avg, margin_error.avg, binary_margin_error.avg,
                        auc*100., fpr_98*100., fpr_991*100., fpr_993*100., fpr_995*100., fpr_997*100., fpr_999*100., fpr_1*100.,
                        acc_class, loss_class, acc_bi_class, loss_bi_class, me_class, me_bi_class)

    return auc
    


if __name__ == '__main__':
    main()
