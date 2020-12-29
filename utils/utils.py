import os
import math

from sklearn import metrics
import numpy as np

import torch
import torch.nn as nn
import torch.nn.functional as F

from torch.autograd import Variable
from torchvision import transforms



def save_checkpoint(save_dir, state, is_best, filename=None):
    """
    Save the latest and best training model
    """
    filename = 'checkpoint_latest.tar' if filename is None else filename
    filename = os.path.join(save_dir, filename)
    torch.save(state, filename)
    if is_best:
        filename = os.path.join(save_dir, 'checkpoint_best.tar')
        torch.save(state, filename)


def accuracy(output, target, topk=(1,)):
    """Computes the precision@k for the specified values of k"""
    maxk = max(topk)
    batch_size = target.size(0)

    _, pred = output.topk(maxk, 1, True, True)
    pred = pred.t()
    correct = pred.eq(target.view(1, -1).expand_as(pred))

    res = []
    for k in topk:
        correct_k = correct[:k].view(-1).float().sum(0)
        res.append(correct_k.mul_(100.0 / batch_size))
    return res

class AverageMeter(object):
    """Computes and stores the average and current value"""
    def __init__(self):
        self.reset()

    def reset(self):
        self.val = 0
        self.avg = 0
        self.sum = 0
        self.count = 0

    def update(self, val, n=1):
        self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count
        
class AUCMeter(object):
    """Computes and stores AUC"""
    def __init__(self):
        self.reset()
        
    def reset(self):
        self.y_true = []
        self.y_score = []
        self.image_ids = []
        
    def update(self, y_true, y_score, image_id):
        self.y_true.append(y_true)
        self.y_score.append(y_score)
        self.image_ids.append(image_id)
        
    def calculate(self):
        y_true = torch.cat(self.y_true)
        y_score = torch.cat(self.y_score)
        auc = metrics.roc_auc_score(y_true!=4, 1.-y_score[:,4])
        fpr, tpr, thresholds = metrics.roc_curve(y_true!=4, 1.-y_score[:,4])
        fpr_991 = fpr[np.where(tpr>=0.991)[0][0]]
        fpr_993 = fpr[np.where(tpr>=0.993)[0][0]]
        fpr_995 = fpr[np.where(tpr>=0.995)[0][0]]
        fpr_997 = fpr[np.where(tpr>=0.997)[0][0]]
        fpr_999 = fpr[np.where(tpr>=0.999)[0][0]]
        fpr_1 = fpr[np.where(tpr==1.)[0][0]]
        
        return auc, fpr_991, fpr_993, fpr_995, fpr_997, fpr_999, fpr_1
    
class Timeline(object):
    """Stores epoch values"""
    def __init__(self):
        self.loss = []
        self.acc = []
        self.loss_bi = []
        self.acc_bi = []
        self.margin_error = []
        self.margin_error_bi = []
        self.auc = []
        self.fpr_991 = []
        self.fpr_993 = []
        self.fpr_995 = []
        self.fpr_997 = []
        self.fpr_999 = []
        self.fpr_1 = []
        self.acc_class = []
        self.loss_class = []
        self.acc_bi_class = []
        self.loss_bi_class = []
        self.me_class = []
        self.me_bi_class = []
        
    def update(self, loss, acc, loss_bi, acc_bi, margin_error, margin_error_bi, auc, fpr_991, fpr_993, fpr_995, fpr_997, fpr_999, fpr_1, acc_class, loss_class, acc_bi_class, loss_bi_class, me_class, me_bi_class):
        self.loss.append(loss)
        self.acc.append(acc)
        self.loss_bi.append(loss_bi)
        self.acc_bi.append(acc_bi)
        self.margin_error.append(margin_error)
        self.margin_error_bi.append(margin_error_bi)
        self.auc.append(auc)
        self.fpr_991.append(fpr_991)
        self.fpr_993.append(fpr_993)
        self.fpr_995.append(fpr_995)
        self.fpr_997.append(fpr_997)
        self.fpr_999.append(fpr_999)
        self.fpr_1.append(fpr_1)
        self.acc_class.append(acc_class)
        self.loss_class.append(loss_class)
        self.acc_bi_class.append(acc_bi_class)
        self.loss_bi_class.append(loss_bi_class)
        self.me_class.append(me_class)
        self.me_bi_class.append(me_bi_class)

def get_margin(output, label):
    top2 = output.topk(2)[1]
    pred = torch.argmax(output, dim=1)
    error_idx = (label != pred)
    margin = torch.zeros(len(output)).cuda()

    margin[error_idx] = output[error_idx, label[error_idx]] - output[error_idx, pred[error_idx]]
    margin[~error_idx] = output[~error_idx, label[~error_idx]] - output[~error_idx, top2[~error_idx][:, 1]]
    return margin

def ramp_loss(margin, gamma):
    loss = 1 - margin / gamma
    loss[np.where(margin > gamma)[0]] = 0
    loss[np.where(margin < 0)[0]] = 1
    return loss.mean()

def margin_error(margin, gamma):
    return np.where(margin < gamma, 1, 0).mean()

def l2normalize(v, eps=1e-6):
    return v/(v.norm() + eps)

def get_spectral(w, iters, u, v):
    ## w should be detached
    height = w.shape[0]
    width = w.view(height, -1).shape[1]

    if u is None:
        u = torch.randn(height).cuda()
        v = torch.randn(width).cuda()
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)

    with torch.no_grad():
        for _ in range(iters):
            v.data = l2normalize(torch.mv(torch.t(w.view(height,-1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height,-1).data, v.data))

        sig = u.dot(w.view(height, -1).mv(v)).item()

    return sig, u, v

def get_21norm(w):
    height = w.shape[0] ##output channel
    norm_2 = w.view(height, -1).norm(p=2, dim=0)
    norm_21 = norm_2.norm(p=1).item()
    return norm_21

def get_12norm(w):
    height = w.shape[0] ##output channel
    norm_2 = w.view(height, -1).norm(p=1, dim=0)
    norm_21 = norm_2.norm(p=2).item()
    return norm_21

def get_Lip_pow(resnet, u, v, block='BasicBlock'):
    ## u & v are dictionary

    def get_MidLip(conv, bn, u, v):
        size = conv.weight.size()
        w_cnn = conv.weight.view(size[0], -1)
        scale = bn.weight / (bn.running_var.sqrt() + bn.eps)
        w_cnn_new = w_cnn * scale.view(-1, 1)
        pow_iters = 30 if u is None else 8
        return get_spectral(w_cnn_new.data, pow_iters, u, v)

    ## BasicBlock or Bottleneck
    lip = 1
    ## conv-1 & bn-1
    w_norm, u['c1'], v['c1'] = get_MidLip(resnet.conv1, resnet.bn1, u['c1'], v['c1'])
    lip *= w_norm

    for m_, module in enumerate([resnet.layer1, resnet.layer2,
                                 resnet.layer3, resnet.layer4]):
        # module is nn.Sequential() object with num of block len(module)
        lip_mod = 1
        if block == 'BasicBlock':
            for b_ in range(len(module)):
                w_norm1, u['m%db%dc1'%(m_,b_)], v['m%db%dc1'%(m_,b_)] = get_MidLip(module[b_].conv1, module[b_].bn1, 
                                                                u['m%db%dc1'%(m_,b_)], v['m%db%dc1'%(m_,b_)])
                w_norm2, u['m%db%dc2'%(m_,b_)], v['m%db%dc2'%(m_,b_)] = get_MidLip(module[b_].conv2, module[b_].bn2, 
                                                            u['m%db%dc2'%(m_,b_)], v['m%db%dc2'%(m_,b_)])                
                if len(list(zip(*module[b_].shortcut.named_children()))) != 0:
                    w_norm3, u['m%db%dc3'%(m_,b_)], v['m%db%dc3'%(m_,b_)] = get_MidLip(module[b_].shortcut[0], module[b_].shortcut[1], 
                                                                u['m%db%dc3'%(m_,b_)], v['m%db%dc3'%(m_,b_)])
                    lip_block = (math.sqrt(w_norm1*w_norm2) + w_norm3)/10
                else:
                    lip_block = 1 ## need more careful analysis here!
                lip_mod *= lip_block
        lip *= lip_mod

    pow_iters = 30 if u['w_fc'] is None else 8
    w_norm, u['w_fc'], v['w_fc'] = get_spectral(resnet.linear.weight.data, iters=pow_iters,
                                                u=u['w_fc'], v=v['w_fc'])
    lip *= w_norm
    return lip, u, v

def get_Lip_L1(resnet, block='BasicBlock',s=1000):

    def get_MidLip(conv, bn, s):
        size = conv.weight.size()
        w_cnn = conv.weight.view(size[0], -1)
        scale = bn.weight / (bn.running_var.sqrt() + bn.eps)
        w_cnn_new = w_cnn * scale.view(-1, 1)
        return w_cnn_new.norm(1).item() / s

    ## BasicBlock or Bottleneck
    lip = 1
    ## conv-1 & bn-1
    lip *= get_MidLip(resnet.conv1, resnet.bn1, s=s)

    for module in [resnet.layer1, resnet.layer2,
                   resnet.layer3, resnet.layer4]:
        ## module is nn.Sequential() object with num of block len(module)
        lip_mod = 1
        if block == 'BasicBlock':
            for i in range(len(module)):
                lip_block = 1
                lip_block *= get_MidLip(module[i].conv1, module[i].bn1, s=1)
                lip_block *= get_MidLip(module[i].conv2, module[i].bn2, s=1)
                if len(list(zip(*module[i].shortcut.named_children()))) != 0:
                    lip_block += get_MidLip(module[i].shortcut[0],
                                            module[i].shortcut[1], s=1)
                lip_mod *= (lip_block/s)
        lip *= lip_mod
    
    lip *= resnet.linear.weight.norm(2).item() / s
    return lip
