import torch
import torch.nn as nn
from torch.nn import functional as F

import pandas as pd
import numpy as np


class CrossEntropy():
    def __init__(self, labels, num_epochs, num_classes=10):
        self.crit = nn.CrossEntropyLoss()
        
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.true_labels = pd.read_csv('/home/kaiyihuang/nexperia/new_data/true_labels.csv', index_col=0)
        self.clean_labels = pd.read_csv('/home/kaiyihuang/nexperia/clean_labels.csv', index_col=0)
        self.image_id_index = pd.read_csv('/home/kaiyihuang/nexperia/image_id_index.csv', index_col=0)
        self.weights = torch.zeros(num_epochs, len(self.true_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.clean_weights = torch.zeros(num_epochs, len(self.clean_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)

<<<<<<< HEAD
    def __call__(self, logits, targets, index, epoch, state='train', mod=None):
=======
    def __call__(self, logits, targets, index, epoch, state='train'):
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
        loss =  self.crit(logits, targets)
        
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)
        loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float())
        margin_error = torch.mean(prob[np.arange(len(targets)),targets]
                                  - torch.max(
                                      prob[
                                          torch.arange(prob.size(1)).reshape(1,-1).repeat(len(targets),1)
                                          !=targets.reshape(-1,1).repeat(1,prob.size(1)).cpu()]
                                      .view(len(targets), -1), 1)[0])
        margin_error_bi = torch.mean((prob[:,4] * 2 - 1) * torch.sign((targets==4).int() - 0.5))

        if state=='val':
            index+=31025
        elif state=='test':
            index+=34472
        elif state!='train':
            raise KeyError("State {} is not supported.".format(state))
        
        self.soft_labels[index] = prob
        
        return loss, loss_bi, margin_error, margin_error_bi


class CrossEntropyWeightedBinary():
<<<<<<< HEAD
    def __init__(self, labels, num_epochs, num_classes=10,
                 el1=None, el2=None, el3=None, el4=None, el5=None,
                 el6=None, el7=None, el8=None, el9=None, el10=None, momentum=1):
=======
    def __init__(self, labels, num_epochs, num_classes=10, el1=None, el2=None, el3=None, el4=None, el5=None, el6=None, el7=None, el8=None, el9=None, el10=None, momentum=1):
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
        self.crit = nn.CrossEntropyLoss(reduction='none')
        
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.true_labels = pd.read_csv('/home/kaiyihuang/nexperia/new_data/true_labels.csv', index_col=0)
        self.clean_labels = pd.read_csv('/home/kaiyihuang/nexperia/clean_labels.csv', index_col=0)
        self.image_id_index = pd.read_csv('/home/kaiyihuang/nexperia/image_id_index.csv', index_col=0)
        self.weights = torch.zeros(num_epochs, len(self.true_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.clean_weights = torch.zeros(num_epochs, len(self.clean_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)
        
        self.el = torch.tensor([el1, el2, el3, el4, el5, el6, el7, el8, el9, el10]).cuda(non_blocking=True)
        self.momentum = momentum

<<<<<<< HEAD
    def __call__(self, logits, targets, index, epoch, state='train', mod=None):
=======
    def __call__(self, logits, targets, index, epoch, state='train'):
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
        loss = self.crit(logits, targets)
        crit = nn.CrossEntropyLoss()
        
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)
        loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float(), reduction='none')
        
        if state=='train':
            binary_weight = (epoch >= self.el[targets]).float() * self.momentum
            loss = (loss + binary_weight * loss_bi) / (1 + binary_weight)            
        
        margin_error = torch.mean(prob[np.arange(len(targets)),targets]
                                  - torch.max(
                                      prob[
                                          torch.arange(prob.size(1)).reshape(1,-1).repeat(len(targets),1)
                                          !=targets.reshape(-1,1).repeat(1,prob.size(1)).cpu()]
                                      .view(len(targets), -1), 1)[0])
        margin_error_bi = torch.mean((prob[:,4] * 2 - 1) * torch.sign((targets==4).int() - 0.5))

        if state=='val':
            index+=31025
        elif state=='test':
            index+=34472
        elif state!='train':
            raise KeyError("State {} is not supported.".format(state))
        
        self.soft_labels[index] = prob
        
        return torch.mean(loss), torch.mean(loss_bi), margin_error, margin_error_bi


class SelfAdaptiveTrainingCE():
    def __init__(self, labels, num_epochs, num_classes=10, momentum=0.9, es=40):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.outputs = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.outputs[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es
        
        self.true_labels = pd.read_csv('/home/kaiyihuang/nexperia/new_data/true_labels.csv', index_col=0)
        self.clean_labels = pd.read_csv('/home/kaiyihuang/nexperia/clean_labels.csv', index_col=0)
        self.image_id_index = pd.read_csv('/home/kaiyihuang/nexperia/image_id_index.csv', index_col=0)
        self.weights = torch.zeros(num_epochs, len(self.true_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.clean_weights = torch.zeros(num_epochs, len(self.clean_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, logits, targets, index, epoch, state='train', mod=None):
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)

        margin_error = torch.mean(prob[np.arange(len(targets)),targets]
                                  - torch.max(
                                      prob[
                                          torch.arange(prob.size(1)).reshape(1,-1).repeat(len(targets),1)
                                          !=targets.reshape(-1,1).repeat(1,prob.size(1)).cpu()]
                                      .view(len(targets), -1), 1)[0])
        margin_error_bi = torch.mean((prob[:,4] * 2 - 1) * torch.sign((targets==4).int() - 0.5))

        if state=='val':
            index+=31025
        elif state=='test':
            index+=34472
        elif state!='train':
            raise KeyError("State {} is not supported.".format(state))

        self.outputs[index] = prob
        if epoch < self.es:
            loss = F.cross_entropy(logits, targets)
            loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float())
            return loss, loss_bi, margin_error, margin_error_bi
        else:
            if mod is not None:
                self.soft_labels[index[targets==4]] = self.momentum*self.soft_labels[index[targets==4]]+(1-self.momentum)*prob[targets==4]
            else:
                self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        if state=='train':
            # obtain weights
            weights, _ = self.soft_labels[index].max(dim=1)
<<<<<<< HEAD
            if mod=='bad_boost':
                weights[
                    torch.logical_or(torch.logical_or(targets==5, targets==7),
                                     torch.logical_or(targets==8, targets==9))] = 5.
=======
            if mod=='bad_1':
                weights[targets!=4] = 1.
            if mod=='bad_boost':
                weights[targets!=4] = 1.
                weights[np.logical_or(np.logical_or(targets==5, targets==7), np.logical_or(targets==8, targets==9))]=5.
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
            weights *= logits.shape[0] / weights.sum()

            # compute cross entropy loss, without reduction
            loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)

            # sample weighted mean
            loss = (loss * weights).mean()
            loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float(), weight=weights)
        else:
            loss = F.cross_entropy(logits, targets)
            loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float())
        
        return loss, loss_bi, margin_error, margin_error_bi


<<<<<<< HEAD
class SelfAdaptiveTrainingCEMultiWeightedBCE():
    def __init__(self, labels, num_epochs, num_classes=10,
                 es1=None, es2=None, es3=None, es4=None, es5=None,
                 es6=None, es7=None, es8=None, es9=None, es10=None,
                 el1=None, el2=None, el3=None, el4=None, el5=None,
                 el6=None, el7=None, el8=None, el9=None, el10=None, ce_momentum=1, momentum=0.9):        
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.outputs = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.outputs[torch.arange(labels.shape[0]), labels] = 1
        self.es = torch.tensor([es1, es2, es3, es4, es5, es6, es7, es8, es9, es10]).cuda(non_blocking=True)
        self.el = torch.tensor([el1, el2, el3, el4, el5, el6, el7, el8, el9, el10]).cuda(non_blocking=True)
        self.ce_momentum = ce_momentum
        self.momentum = momentum
        
        self.true_labels = pd.read_csv('/home/kaiyihuang/nexperia/new_data/true_labels.csv', index_col=0)
        self.clean_labels = pd.read_csv('/home/kaiyihuang/nexperia/clean_labels.csv', index_col=0)
        self.image_id_index = pd.read_csv('/home/kaiyihuang/nexperia/image_id_index.csv', index_col=0)
        self.weights = torch.zeros(num_epochs, len(self.true_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.clean_weights = torch.zeros(num_epochs, len(self.clean_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, logits, targets, index, epoch, state='train', mod=None):
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)

        margin_error = torch.mean(prob[np.arange(len(targets)),targets]
                                  - torch.max(
                                      prob[
                                          torch.arange(prob.size(1)).reshape(1,-1).repeat(len(targets),1)
                                          !=targets.reshape(-1,1).repeat(1,prob.size(1)).cpu()]
                                      .view(len(targets), -1), 1)[0])
        margin_error_bi = torch.mean((prob[:,4] * 2 - 1) * torch.sign((targets==4).int() - 0.5))

        if state=='val':
            index+=31025
        elif state=='test':
            index+=34472
        elif state!='train':
            raise KeyError("State {} is not supported.".format(state))

        self.outputs[index] = prob
        momentum = 1 - (1 - self.momentum) * (epoch>=self.es[targets])
        self.soft_labels[index] = torch.mul(
            self.soft_labels[index], momentum.view(-1,1)) + torch.mul(prob, (1-momentum).view(-1,1))

        if state=='train':
            # obtain weights
            weights, _ = self.soft_labels[index].max(dim=1)
            weights *= logits.shape[0] / weights.sum()

            # compute cross entropy loss, without reduction
            loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)

            # sample weighted mean
            loss = (loss * weights).mean()
            loss_bi = -((1-self.soft_labels[index][:,4]) * torch.log(1-prob[:,4]) \
                        + self.soft_labels[index][:,4] * torch.log(prob[:,4])) * weights
            
            binary_weight = (epoch >= self.el[targets]).float() * self.ce_momentum
            loss = (loss + binary_weight * loss_bi) / (1 + binary_weight)
            loss = torch.mean(loss)
            loss_bi = torch.mean(loss_bi)

        else:
            loss = F.cross_entropy(logits, targets)
            loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float())
        
        return loss, loss_bi, margin_error, margin_error_bi


class SelfAdaptiveTrainingWeightedBCE():
    def __init__(self, labels, num_epochs, num_classes=10,
                 el1=None, el2=None, el3=None, el4=None, el5=None,
                 el6=None, el7=None, el8=None, el9=None, el10=None, ce_momentum=1, momentum=0.9, es=40):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.outputs = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.outputs[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es
        self.el = torch.tensor([el1, el2, el3, el4, el5, el6, el7, el8, el9, el10]).cuda(non_blocking=True)
        self.ce_momentum = ce_momentum
        
        self.true_labels = pd.read_csv('/home/kaiyihuang/nexperia/new_data/true_labels.csv', index_col=0)
        self.clean_labels = pd.read_csv('/home/kaiyihuang/nexperia/clean_labels.csv', index_col=0)
        self.image_id_index = pd.read_csv('/home/kaiyihuang/nexperia/image_id_index.csv', index_col=0)
        self.weights = torch.zeros(num_epochs, len(self.true_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.clean_weights = torch.zeros(num_epochs, len(self.clean_labels), num_classes, dtype=torch.float).cuda(non_blocking=True)

    def __call__(self, logits, targets, index, epoch, state='train', mod=None):
        # obtain prob, then update running avg
        prob = F.softmax(logits.detach(), dim=1)

        margin_error = torch.mean(prob[np.arange(len(targets)),targets]
                                  - torch.max(
                                      prob[
                                          torch.arange(prob.size(1)).reshape(1,-1).repeat(len(targets),1)
                                          !=targets.reshape(-1,1).repeat(1,prob.size(1)).cpu()]
                                      .view(len(targets), -1), 1)[0])
        margin_error_bi = torch.mean((prob[:,4] * 2 - 1) * torch.sign((targets==4).int() - 0.5))

        if state=='val':
            index+=31025
        elif state=='test':
            index+=34472
        elif state!='train':
            raise KeyError("State {} is not supported.".format(state))

        self.outputs[index] = prob
        if epoch < self.es:
            loss = F.cross_entropy(logits, targets)
            loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float())
            return loss, loss_bi, margin_error, margin_error_bi
        else:
            if mod is not None:
                self.soft_labels[index[targets==4]] = self.momentum*self.soft_labels[index[targets==4]]+(1-self.momentum)*prob[targets==4]
            else:
                self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob

        if state=='train':
            # obtain weights
            weights, _ = self.soft_labels[index].max(dim=1)
            if mod=='bad_boost':
                weights[
                    torch.logical_or(
                        torch.logical_or(targets==5, targets==7),
                        torch.logical_or(targets==8, targets==9))] = 5.
            weights *= logits.shape[0] / weights.sum()

            # compute cross entropy loss, without reduction
            loss = torch.sum(-F.log_softmax(logits, dim=1) * self.soft_labels[index], dim=1)

            # sample weighted mean
            loss = loss * weights
            loss_bi = -((1-self.soft_labels[index][:,4]) * torch.log(1-prob[:,4]) \
                        + self.soft_labels[index][:,4] * torch.log(prob[:,4])) * weights
            
            binary_weight = (epoch >= self.el[targets]).float() * self.ce_momentum
            loss = (loss + binary_weight * loss_bi) / (1 + binary_weight)
            loss = torch.mean(loss)
            loss_bi = torch.mean(loss_bi)

        else:
            loss = F.cross_entropy(logits, targets)
            loss_bi = F.binary_cross_entropy(1-prob[:,4], (targets!=4).float())
            
        
        
        return loss, loss_bi, margin_error, margin_error_bi


=======
>>>>>>> 35ae2811a1414d2aa5319e131a62636cf49648fb
class SelfAdaptiveTrainingSCE():
    def __init__(self, labels, num_classes=10, momentum=0.9, es=40, alpha=1, beta=0.3):
        # initialize soft labels to onthot vectors
        self.soft_labels = torch.zeros(labels.shape[0], num_classes, dtype=torch.float).cuda(non_blocking=True)
        self.soft_labels[torch.arange(labels.shape[0]), labels] = 1
        self.momentum = momentum
        self.es = es
        self.alpha = alpha
        self.beta = beta
        print("alpha = {}, beta = {}".format(alpha, beta))


    def __call__(self, logits, targets, index, epoch):
        if epoch < self.es:
            return F.cross_entropy(logits, targets)

        # obtain prob, then update running avg
        prob = F.softmax(logits, dim=1)
        self.soft_labels[index] = self.momentum * self.soft_labels[index] + (1 - self.momentum) * prob.detach()

        # obtain weights based largest and second largest prob
        weights, _ = self.soft_labels[index].max(dim=1)
        weights *= logits.shape[0] / weights.sum()

        # use symmetric cross entropy loss, without reduction
        loss = - self.alpha * torch.sum(self.soft_labels[index] * torch.log(prob), dim=-1) \
                - self.beta * torch.sum(prob * torch.log(self.soft_labels[index]), dim=-1)

        # sample weighted mean
        loss = (loss * weights).mean()
        return loss
