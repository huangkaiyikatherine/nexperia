from __future__ import absolute_import

from .loss import CrossEntropy, CrossEntropyWeightedBinary, SelfAdaptiveTrainingCE, SelfAdaptiveTrainingCEMultiWeightedBCE, SelfAdaptiveTrainingWeightedBCE, SelfAdaptiveTrainingSCE, WeightedCrossEntropy, CrossEntropyGeneral, CrossEntropyTrain, SelfAdaptiveTrainingCEGeneral, FocalLoss, FocalLossGeneral, SelfAdaptiveTrainingFL, SelfAdaptiveTrainingFLGeneral, SelfAdaptiveTrainingCETrain, FocalLossTrain, SelfAdaptiveTrainingFLTrain

from .trades import TRADES, TRADES_SAT


def get_loss(args, labels=None, num_classes=10, datasets=None, train_len=None, val_len=None, test_len=None, pass_idx=None):
    if args.loss == 'ce':
        if args.dataset=='nexperia_merge':
            criterion = CrossEntropyGeneral(len(datasets['train']), len(datasets['val']), len(datasets['test']),
                                     num_epochs=args.epochs, num_classes=num_classes)
        elif args.dataset=='nexperia_train' or args.dataset=='nexperia_eval':
            criterion = CrossEntropyTrain(labels, train_len, val_len, test_len, pass_idx,
                                     num_epochs=args.epochs, num_classes=num_classes)
        else:
            criterion = CrossEntropy(labels, num_classes=num_classes, num_epochs=args.epochs)
        
    elif args.loss == 'wce':
        criterion = WeightedCrossEntropy(labels, num_classes=num_classes, num_epochs=args.epochs)
    
    elif args.loss == 'sat':
        if args.dataset=='nexperia_merge':
            criterion = SelfAdaptiveTrainingCEGeneral(len(datasets['train']), len(datasets['val']), len(datasets['test']),
                                                      num_epochs=args.epochs, num_classes=num_classes,
                                                      momentum=args.sat_alpha, es=args.sat_es)
        elif args.dataset=='nexperia_train' or args.dataset=='nexperia_eval':
            criterion = SelfAdaptiveTrainingCETrain(labels, train_len, val_len, test_len, pass_idx,
                                                    num_classes=num_classes, momentum=args.sat_alpha,
                                                    es=args.sat_es, num_epochs=args.epochs)
        else:
            criterion = SelfAdaptiveTrainingCE(
                labels, num_classes=num_classes, momentum=args.sat_alpha, es=args.sat_es, num_epochs=args.epochs)
    
    elif args.loss == 'fl':
        if args.dataset=='nexperia_merge':
            criterion = FocalLossGeneral(len(datasets['train']), len(datasets['val']), len(datasets['test']),
                                     num_epochs=args.epochs, num_classes=num_classes, alpha=args.fl_alpha, gamma=args.fl_gamma)
        elif args.dataset=='nexperia_train' or args.dataset=='nexperia_eval':
            criterion = FocalLossTrain(labels, train_len, val_len, test_len, pass_idx,
                                     num_epochs=args.epochs, num_classes=num_classes, alpha=args.fl_alpha, gamma=args.fl_gamma)
        else:
            criterion = FocalLoss(labels, num_classes=num_classes, num_epochs=args.epochs, alpha=args.fl_alpha, gamma=args.fl_gamma)
    
    elif args.loss == 'sat_fl':
        if args.dataset=='nexperia_merge':
            criterion = SelfAdaptiveTrainingFLGeneral(len(datasets['train']), len(datasets['val']), len(datasets['test']),
                                                      num_epochs=args.epochs, num_classes=num_classes,
                                                      momentum=args.sat_alpha, es=args.sat_es,
                                                      lamb=args.fl_lambda, alpha=args.fl_alpha, gamma=args.fl_gamma)
        elif args.dataset=='nexperia_train' or args.dataset=='nexperia_eval':
            criterion = SelfAdaptiveTrainingFLTrain(labels, train_len, val_len, test_len, pass_idx,
                                                    num_epochs=args.epochs, num_classes=num_classes,
                                                    momentum=args.sat_alpha, es=args.sat_es,
                                                    lamb=args.fl_lambda, alpha=args.fl_alpha, gamma=args.fl_gamma)
        else:
            criterion = SelfAdaptiveTrainingFL(labels, num_classes=num_classes, momentum=args.sat_alpha, es=args.sat_es,
                                               num_epochs=args.epochs, lamb=args.fl_lambda, alpha=args.fl_alpha, gamma=args.fl_gamma)
    
    elif args.loss == 'sat_sce':
        alpha, beta = 1, 0.3
        criterion = SelfAdaptiveTrainingSCE(labels, num_classes=num_classes, momentum=args.sat_alpha, es=args.sat_es, alpha=alpha, beta=beta)
    
    elif args.loss == 'trades':
        criterion = TRADES(step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps, beta=args.beta)
    
    elif args.loss == 'trades_sat':
        criterion = TRADES_SAT(labels, num_classes=num_classes, momentum=args.sat_alpha , es=args.sat_es,
                        step_size=args.step_size, epsilon=args.epsilon, perturb_steps=args.num_steps, beta=args.beta)
        
    elif args.loss == 'binary_weighted_ce':
        criterion = CrossEntropyWeightedBinary(labels, num_classes=num_classes, num_epochs=args.epochs,
                                               el1=args.el1, el2=args.el2, el3=args.el3, el4=args.el4, el5=args.el5,
                                               el6=args.el6, el7=args.el7, el8=args.el8, el9=args.el9, el10=args.el10,
                                               momentum=args.ce_momentum)
    elif args.loss == 'sat_binary_weighted_ce':
        criterion = SelfAdaptiveTrainingWeightedBCE(labels, num_classes=num_classes, num_epochs=args.epochs,
                                                    el1=args.el1, el2=args.el2, el3=args.el3, el4=args.el4, el5=args.el5,
                                                    el6=args.el6, el7=args.el7, el8=args.el8, el9=args.el9, el10=args.el10,
                                                    ce_momentum=args.ce_momentum, momentum=args.sat_alpha, es=args.sat_es)
    elif args.loss == 'sat_multi_es_weighted_ce':
        criterion = SelfAdaptiveTrainingCEMultiWeightedBCE(labels, num_classes=num_classes, num_epochs=args.epochs,
                                                           es1=args.sat_es1, es2=args.sat_es2, es3=args.sat_es3,
                                                           es4=args.sat_es4, es5=args.sat_es5, es6=args.sat_es6,
                                                           es7=args.sat_es7, es8=args.sat_es8, es9=args.sat_es9,
                                                           es10=args.sat_es10,
                                                           el1=args.el1, el2=args.el2, el3=args.el3, el4=args.el4, el5=args.el5,
                                                           el6=args.el6, el7=args.el7, el8=args.el8, el9=args.el9, el10=args.el10,
                                                           ce_momentum=args.ce_momentum, momentum=args.sat_alpha)
    else:
        raise KeyError("Loss `{}` is not supported.".format(args.loss))

    return criterion
