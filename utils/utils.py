import os
import torch
from sklearn import metrics
import numpy as np


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

