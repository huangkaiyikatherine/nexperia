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
    ## dynamically adjust hyper-parameters for ResNets according to base_width
    if args.base_width != 64 and 'sat' in args.loss:
        factor = 64. / args.base_width
        args.sat_alpha = args.sat_alpha**(1. / factor)
        args.sat_es = int(args.sat_es * factor)
        print("Adaptive parameters adjustment: alpha = {:.3f}, Es = {:d}".format(args.sat_alpha, args.sat_es))

    print(args)
    global best_prec1
    global best_auc_1, best_auc_2, best_auc_3, best_model_1, best_model_2, best_model_3, best_epoch_1, best_epoch_2, best_epoch_3

    # Check the save_dir exists or not
    if not os.path.exists(args.save_dir):
        os.makedirs(args.save_dir)
        os.makedirs(os.path.join(args.save_dir, 'train'))
        os.makedirs(os.path.join(args.save_dir, 'val'))
        os.makedirs(os.path.join(args.save_dir, 'test'))

    # prepare dataset
    train_loader, val_loaders, test_loader, num_classes, train_targets, val_targets, test_targets, targets, pass_idx = get_loader(args)
    
    model = get_model(args, num_classes, base_width=args.base_width)
    if torch.cuda.device_count() > 1:
        model = nn.DataParallel(model)
    model.cuda()

    # optionally resume from a checkpoint
    if args.resume:
        if os.path.isfile(args.resume):
            print("=> loading checkpoint '{}'".format(args.resume))
            checkpoint = torch.load(args.resume)
            args.start_epoch = checkpoint['epoch']
            if args.dataset=='nexperia_split':
                best_auc = checkpoint['best_auc']
            else:
                best_prec1 = checkpoint['best_prec1']
            model.load_state_dict(checkpoint['state_dict'])
            print("=> loaded checkpoint '{}' (epoch {})"
                  .format(args.resume, checkpoint['epoch']))
        else:
            print("=> no checkpoint found at '{}'".format(args.resume))

    torch.cuda.manual_seed(args.seed)
    cudnn.benchmark = True

    criterion = get_loss(args, labels=targets, num_classes=num_classes,
                         train_len=len(train_targets), val_len=len(val_targets), test_len=len(test_targets),
                         pass_idx=pass_idx)
    optimizer = get_optimizer(model, args)
    scheduler = get_scheduler(optimizer, args)
    
    train_timeline = Timeline()
    val_timeline = Timeline()
    test_timeline = Timeline()
    
    if args.evaluate:
        validate(test_loader, model, 0, val_timeline, args.dataset, criterion=criterion, crop=args.crop)
        return

    print("*" * 40)
    start = time.time()
    for epoch in range(args.start_epoch, args.epochs):

        # train for one epoch
        train(train_loader, model, criterion, optimizer, epoch, train_timeline, args.sat_es, args.dataset, args.mod, args.crop)
        print("*" * 40)
        
        scheduler.step()

        # evaluate on validation sets
        prec1 = 0
        print('val:')
        val_auc = validate(
            val_loaders, model, epoch, val_timeline, args.dataset, state='val', criterion=criterion, crop=args.crop)
        print("*" * 40)

        print('test:')
        test_auc = validate(
            test_loader, model, epoch, test_timeline, args.dataset, state='test', criterion=criterion, crop=args.crop)
        print("*" * 40)

        # remember best auc and save checkpoint
        is_best_1 = val_auc > best_auc_1
        is_best_2 = val_auc > best_auc_2
        is_best_3 = val_auc > best_auc_3
        
        if is_best_1:
            best_auc_3 = best_auc_2
            best_auc_2 = best_auc_1
            best_auc_1 = val_auc
            best_model_3 = copy.deepcopy(best_model_2)
            best_model_2 = copy.deepcopy(best_model_1)
            best_model_1 = copy.deepcopy(model.state_dict())
            best_epoch_3 = best_epoch_2
            best_epoch_2 = best_epoch_1
            best_epoch_1 = epoch
        elif is_best_2:
            best_auc_3 = best_auc_2
            best_auc_2 = val_auc
            best_model_3 = copy.deepcopy(best_model_2)
            best_model_2 = copy.deepcopy(model.state_dict())
            best_epoch_3 = best_epoch_2
            best_epoch_2 = epoch
        elif is_best_3:
            best_auc_3 = val_auc
            best_model_3 = copy.deepcopy(model.state_dict())
            best_epoch_3 = epoch
            

        if args.save_freq > 0 and (epoch + 1) % args.save_freq == 0:
            filename = 'checkpoint_{}.tar'.format(epoch + 1)
        else:
            filename = None
        save_checkpoint(args.save_dir, {
            'epoch': epoch + 1,
            'state_dict': model.state_dict(),
            'best_auc_1': best_auc_1,
            'best_auc_2': best_auc_2,
            'best_auc_3': best_auc_3,
            'best_model_1': best_model_1,
            'best_model_2': best_model_2,
            'best_model_3': best_model_3,
            'best_epoch_1': best_epoch_1,
            'best_epoch_2': best_epoch_2,
            'best_epoch_3': best_epoch_3,
        }, filename=filename)
                            
    # evaludate latest checkpoint
    print("Test acc of latest checkpoint:", end='\t')
    validate(test_loader, model, epoch, test_timeline, args.dataset, criterion=criterion, last=True, crop=args.crop)
    print("*" * 40)

    # evaluate best 3 checkpoints
    checkpoint = torch.load(os.path.join(args.save_dir, 'checkpoint_latest.tar'))
    print("Best 1 validation auc ({}th epoch): {:.2f}%".format(checkpoint['best_epoch_1'], best_auc_1))
    model.load_state_dict(checkpoint['best_model_1'])
    print("Test acc of best 1 checkpoint:", end='\t')
    validate(test_loader, model, checkpoint['best_epoch_1'], test_timeline, args.dataset, criterion=criterion, last=True, crop=args.crop)
    print("*" * 40)

    print("Best 2 validation auc ({}th epoch): {:.2f}%".format(checkpoint['best_epoch_2'], best_auc_2))
    model.load_state_dict(checkpoint['best_model_2'])
    print("Test acc of best 2 checkpoint:", end='\t')
    validate(test_loader, model, checkpoint['best_epoch_2'], test_timeline, args.dataset, criterion=criterion, last=True, crop=args.crop)
    print("*" * 40)

    print("Best 3 validation auc ({}th epoch): {:.2f}%".format(checkpoint['best_epoch_3'], best_auc_3))
    model.load_state_dict(checkpoint['best_model_3'])
    print("Test acc of best 3 checkpoint:", end='\t')
    validate(test_loader, model, checkpoint['best_epoch_3'], test_timeline, args.dataset, criterion=criterion, last=True, crop=args.crop)
    print("*" * 40)

    time_elapsed = time.time() - start
    print('It takes {:.0f}m {:.0f}s to train.'.format(time_elapsed // 60, time_elapsed % 60))
    
    # save soft label
    if hasattr(criterion, 'soft_labels'):
        out_fname = os.path.join(args.save_dir, 'updated_soft_labels.npy')
        np.save(out_fname, criterion.soft_labels.cpu().numpy())
        print("Updated soft labels is saved to {}".format(out_fname))
        
    # save weights change of 106 images
    if hasattr(criterion, 'weights'):
        out_fname = os.path.join(args.save_dir, 'weights_change.npy')
        np.save(out_fname, criterion.weights.cpu().numpy())
        print("weights change is saved to {}".format(out_fname))
        
    if hasattr(criterion, 'clean_weights'):
        out_fname = os.path.join(args.save_dir, 'clean_weights_change.npy')
        np.save(out_fname, criterion.clean_weights.cpu().numpy())
        print("clean weights change is saved to {}".format(out_fname))

    # save timelines
    train_acc_class = torch.cat(train_timeline.acc_class, dim=0)
    train_loss_class = torch.cat(train_timeline.loss_class, dim=0)
    train_acc_bi_class = torch.cat(train_timeline.acc_bi_class, dim=0)
    train_loss_bi_class = torch.cat(train_timeline.loss_bi_class, dim=0)
    train_me_class = torch.cat(train_timeline.me_class, dim=0)
    train_me_bi_class = torch.cat(train_timeline.me_bi_class, dim=0)
        
    val_acc_class = torch.cat(val_timeline.acc_class, dim=0)
    val_loss_class = torch.cat(val_timeline.loss_class, dim=0)
    val_acc_bi_class = torch.cat(val_timeline.acc_bi_class, dim=0)
    val_loss_bi_class = torch.cat(val_timeline.loss_bi_class, dim=0)
    val_me_class = torch.cat(val_timeline.me_class, dim=0)
    val_me_bi_class = torch.cat(val_timeline.me_bi_class, dim=0)
    
    test_acc_class = torch.cat(test_timeline.acc_class, dim=0)
    test_loss_class = torch.cat(test_timeline.loss_class, dim=0)
    test_acc_bi_class = torch.cat(test_timeline.acc_bi_class, dim=0)
    test_loss_bi_class = torch.cat(test_timeline.loss_bi_class, dim=0)
    test_me_class = torch.cat(test_timeline.me_class, dim=0)
    test_me_bi_class = torch.cat(test_timeline.me_bi_class, dim=0)
    
    np.save(os.path.join(args.save_dir, 'train', 'loss.npy'), train_timeline.loss)
    np.save(os.path.join(args.save_dir, 'train', 'acc.npy'), train_timeline.acc)
    np.save(os.path.join(args.save_dir, 'train', 'loss_bi.npy'), train_timeline.loss_bi)
    np.save(os.path.join(args.save_dir, 'train', 'acc_bi.npy'), train_timeline.acc_bi)
    np.save(os.path.join(args.save_dir, 'train', 'loss_class.npy'), train_loss_class)
    np.save(os.path.join(args.save_dir, 'train', 'acc_class.npy'), train_acc_class)
    np.save(os.path.join(args.save_dir, 'train', 'loss_bi_class.npy'), train_loss_bi_class)
    np.save(os.path.join(args.save_dir, 'train', 'acc_bi_class.npy'), train_acc_bi_class)
    np.save(os.path.join(args.save_dir, 'train', 'margin_error.npy'), train_timeline.margin_error)
    np.save(os.path.join(args.save_dir, 'train', 'margin_error_bi.npy'), train_timeline.margin_error_bi)
    np.save(os.path.join(args.save_dir, 'train', 'margin_error_class.npy'), train_me_class)
    np.save(os.path.join(args.save_dir, 'train', 'margin_error_bi_class.npy'), train_me_bi_class)
    np.save(os.path.join(args.save_dir, 'train', 'auc.npy'), train_timeline.auc)
    np.save(os.path.join(args.save_dir, 'train', 'fpr_98.npy'), train_timeline.fpr_98)
    np.save(os.path.join(args.save_dir, 'train', 'fpr_991.npy'), train_timeline.fpr_991)
    np.save(os.path.join(args.save_dir, 'train', 'fpr_993.npy'), train_timeline.fpr_993)
    np.save(os.path.join(args.save_dir, 'train', 'fpr_995.npy'), train_timeline.fpr_995)
    np.save(os.path.join(args.save_dir, 'train', 'fpr_997.npy'), train_timeline.fpr_997)
    np.save(os.path.join(args.save_dir, 'train', 'fpr_999.npy'), train_timeline.fpr_999)
    np.save(os.path.join(args.save_dir, 'train', 'fpr_1.npy'), train_timeline.fpr_1)
    print("other training details are saved to {}".format(os.path.join(args.save_dir, 'train')))

    np.save(os.path.join(args.save_dir, 'val', 'loss.npy'), val_timeline.loss)
    np.save(os.path.join(args.save_dir, 'val', 'acc.npy'), val_timeline.acc)
    np.save(os.path.join(args.save_dir, 'val', 'loss_bi.npy'), val_timeline.loss_bi)
    np.save(os.path.join(args.save_dir, 'val', 'acc_bi.npy'), val_timeline.acc_bi)
    np.save(os.path.join(args.save_dir, 'val', 'loss_class.npy'), val_loss_class)
    np.save(os.path.join(args.save_dir, 'val', 'acc_class.npy'), val_acc_class)
    np.save(os.path.join(args.save_dir, 'val', 'loss_bi_class.npy'), val_loss_bi_class)
    np.save(os.path.join(args.save_dir, 'val', 'acc_bi_class.npy'), val_acc_bi_class)
    np.save(os.path.join(args.save_dir, 'val', 'margin_error.npy'), val_timeline.margin_error_bi)
    np.save(os.path.join(args.save_dir, 'val', 'margin_error_bi.npy'), val_timeline.margin_error_bi)
    np.save(os.path.join(args.save_dir, 'val', 'margin_error_class.npy'), val_me_class)
    np.save(os.path.join(args.save_dir, 'val', 'margin_error_bi_class.npy'), val_me_bi_class)
    np.save(os.path.join(args.save_dir, 'val', 'auc.npy'), val_timeline.auc)
    np.save(os.path.join(args.save_dir, 'val', 'fpr_98.npy'), val_timeline.fpr_98)
    np.save(os.path.join(args.save_dir, 'val', 'fpr_991.npy'), val_timeline.fpr_991)
    np.save(os.path.join(args.save_dir, 'val', 'fpr_993.npy'), val_timeline.fpr_993)
    np.save(os.path.join(args.save_dir, 'val', 'fpr_995.npy'), val_timeline.fpr_995)
    np.save(os.path.join(args.save_dir, 'val', 'fpr_997.npy'), val_timeline.fpr_997)
    np.save(os.path.join(args.save_dir, 'val', 'fpr_999.npy'), val_timeline.fpr_999)
    np.save(os.path.join(args.save_dir, 'val', 'fpr_1.npy'), val_timeline.fpr_1)
    print("other validating details are saved to {}".format(os.path.join(args.save_dir, 'val')))

    np.save(os.path.join(args.save_dir, 'test', 'loss.npy'), test_timeline.loss)
    np.save(os.path.join(args.save_dir, 'test', 'acc.npy'), test_timeline.acc)
    np.save(os.path.join(args.save_dir, 'test', 'loss_bi.npy'), test_timeline.loss_bi)
    np.save(os.path.join(args.save_dir, 'test', 'acc_bi.npy'), test_timeline.acc_bi)
    np.save(os.path.join(args.save_dir, 'test', 'loss_class.npy'), test_loss_class)
    np.save(os.path.join(args.save_dir, 'test', 'acc_class.npy'), test_acc_class)
    np.save(os.path.join(args.save_dir, 'test', 'loss_bi_class.npy'), test_loss_bi_class)
    np.save(os.path.join(args.save_dir, 'test', 'acc_bi_class.npy'), test_acc_bi_class)
    np.save(os.path.join(args.save_dir, 'test', 'margin_error.npy'), test_timeline.margin_error_bi)
    np.save(os.path.join(args.save_dir, 'test', 'margin_error_bi.npy'), test_timeline.margin_error_bi)
    np.save(os.path.join(args.save_dir, 'test', 'margin_error_class.npy'), test_me_class)
    np.save(os.path.join(args.save_dir, 'test', 'margin_error_bi_class.npy'), test_me_bi_class)
    np.save(os.path.join(args.save_dir, 'test', 'auc.npy'), test_timeline.auc)
    np.save(os.path.join(args.save_dir, 'test', 'fpr_98.npy'), test_timeline.fpr_98)
    np.save(os.path.join(args.save_dir, 'test', 'fpr_991.npy'), test_timeline.fpr_991)
    np.save(os.path.join(args.save_dir, 'test', 'fpr_993.npy'), test_timeline.fpr_993)
    np.save(os.path.join(args.save_dir, 'test', 'fpr_995.npy'), test_timeline.fpr_995)
    np.save(os.path.join(args.save_dir, 'test', 'fpr_997.npy'), test_timeline.fpr_997)
    np.save(os.path.join(args.save_dir, 'test', 'fpr_999.npy'), test_timeline.fpr_999)
    np.save(os.path.join(args.save_dir, 'test', 'fpr_1.npy'), test_timeline.fpr_1)
    print("other testing details are saved to {}".format(os.path.join(args.save_dir, 'test')))

def train(train_loader, model, criterion, optimizer, epoch, timeline, es, dataset, mod=None, crop='center'):
    """
        Run one train epoch
    """
    batch_time = AverageMeter()
    data_time = AverageMeter()
    losses = AverageMeter()
    top1 = AverageMeter()
    binary_losses = AverageMeter()
    binary_acc = AverageMeter()
    margin_error = AverageMeter()
    binary_margin_error = AverageMeter()
    
    auc_meter = AUCMeter()

    # switch to train mode
    model.train()

    end = time.time()
    
    for i, (input, target, index, image_id) in enumerate(train_loader):

        # measure data loading time
        data_time.update(time.time() - end)

        input = input.cuda(non_blocking=True)
        target = target.cuda(non_blocking=True)

        # compute output
        if crop!='center':
            bs, ncrops, c, h, w = input.size()
            output = model(input.view(-1, c, h, w)).view(bs, ncrops, -1).mean(1)
        else:
            output = model(input)

        loss, loss_bi, me, margin_error_bi = criterion(output, target, index, epoch, mod=mod)


        # compute gradient and do SGD step
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        output = output.float()
        loss = loss.float()
        loss_bi = loss_bi.float()

        # measure accuracy and record loss
        prec1 = accuracy(output.data, target)[0]

        probs = F.softmax(output, dim=1)
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

        if (i + 1) % args.print_freq == 0 or (i + 1) == len(train_loader):
            lr = optimizer.param_groups[0]['lr']
            print('Epoch: [{0}][{1}/{2}]\t'
                  'LR {lr:.6f}\t'
                  'Time {batch_time.val:.3f} ({batch_time.avg:.3f})\t'
                  'Data {data_time.val:.3f} ({data_time.avg:.3f})\t'
                  'Loss {loss.val:.4f} ({loss.avg:.4f})\t'
                  'Prec@1 {top1.val:.3f} ({top1.avg:.3f})\t'
                  'Binary Loss {loss_bi.val:.4f} ({loss_bi.avg:.4f})\t'
                  'Binary Accuracy {bi_acc.val:.3f} ({bi_acc.avg:.3f})\t'
                  'Margin Error {margin_error.val:.4f} ({margin_error.avg:.4f})\t'
                  'Binary Margin Error {margin_error_bi.val:.4f} ({margin_error_bi.avg:.4f})'.format(
                      epoch+1, i+1, len(train_loader), lr=lr, batch_time=batch_time, data_time=data_time,
                      loss=losses, top1=top1, loss_bi=binary_losses, bi_acc=binary_acc,
                      margin_error=margin_error, margin_error_bi=binary_margin_error))

    acc_class = torch.zeros(1, criterion.num_classes)
    loss_class = torch.zeros(1, criterion.num_classes)
    acc_bi_class = torch.zeros(1, criterion.num_classes)
    loss_bi_class = torch.zeros(1, criterion.num_classes)
    me_class = torch.zeros(1,criterion.num_classes)
    me_bi_class = torch.zeros(1,criterion.num_classes)

    if hasattr(criterion, 'outputs'):
        outputs = criterion.outputs
    else:
        outputs = criterion.soft_labels

    for i in range(criterion.num_classes):
        acc_class[0, i] = torch.sum(torch.argmax(outputs[:criterion.len_train][
            criterion.labels[:criterion.len_train]==i,:], dim=1)==i).item() / np.count_nonzero(
            criterion.labels[:criterion.len_train]==i)
        loss_class[0, i] = -torch.mean(torch.log(outputs[:criterion.len_train][
            criterion.labels[:criterion.len_train]==i,i])).item()
        outputs_class = outputs[:criterion.len_train][criterion.labels[:criterion.len_train]==i]
        me_class[0, i] = torch.mean(outputs_class[:,i] - torch.max(outputs_class[:,np.arange(criterion.num_classes)!=i], dim=1)[0]).item()
        me_bi_class[0, i] = torch.mean((outputs_class[:,criterion.pass_idx] * 2 - 1) * ((i==criterion.pass_idx) - 0.5)) * 2.
        if i!=criterion.pass_idx:
            acc_bi_class[0, i] = torch.sum(outputs[:criterion.len_train][
                criterion.labels[:criterion.len_train]==i,criterion.pass_idx]<=0.5).item() / np.count_nonzero(
                criterion.labels[:criterion.len_train]==i)
            loss_bi_class[0, i] = -torch.mean(torch.log(1-outputs[:criterion.len_train][
                criterion.labels[:criterion.len_train]==i,criterion.pass_idx])).item()
        else:
            acc_bi_class[0, i] = torch.sum(outputs[:criterion.len_train][
                criterion.labels[:criterion.len_train]==i,criterion.pass_idx]>0.5).item() / np.count_nonzero(
                criterion.labels[:criterion.len_train]==i)
            loss_bi_class[0, i] = -torch.mean(torch.log(outputs[:criterion.len_train][
            criterion.labels[:criterion.len_train]==i,criterion.pass_idx])).item()
        
    auc, fpr_98, fpr_991, fpr_993, fpr_995, fpr_997, fpr_999, fpr_1 = auc_meter.calculate(criterion.pass_idx)
    print('Epoch: [{}]\t'
          'AUC: {:.2f}%\t'
          'FPR(=98%, 99.1%, 99.3%, 99.5%, 99.7%, 100%): {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%, {:.2f}%'
          .format(epoch+1, auc*100., fpr_98*100., fpr_991*100., fpr_993*100., fpr_995*100., fpr_997*100., fpr_1*100.))
    
    timeline.update(losses.avg, top1.avg, binary_losses.avg, binary_acc.avg, margin_error.avg, binary_margin_error.avg,
                    auc*100., fpr_98*100., fpr_991*100., fpr_993*100., fpr_995*100., fpr_997*100., fpr_999*100., fpr_1*100.,
                    acc_class, loss_class, acc_bi_class, loss_bi_class, me_class, me_bi_class)


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
