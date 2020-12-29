from __future__ import absolute_import

from .loss import CrossEntropy, CrossEntropyWeightedBinary, SelfAdaptiveTrainingCE, SelfAdaptiveTrainingCEMultiWeightedBCE, SelfAdaptiveTrainingWeightedBCE, SelfAdaptiveTrainingSCE

from .trades import TRADES, TRADES_SAT


def get_loss(args, labels=None, num_classes=10):
    if args.loss == 'ce':
        criterion = CrossEntropy(labels, num_classes=num_classes, num_epochs=args.epochs)
    
    elif args.loss == 'sat':
        criterion = SelfAdaptiveTrainingCE(labels, num_classes=num_classes, momentum=args.sat_alpha, es=args.sat_es, num_epochs=args.epochs)
    
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
