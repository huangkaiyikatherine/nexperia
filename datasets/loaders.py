from __future__ import absolute_import

from copy import deepcopy
import torch
import numpy as np
import pandas as pd

from .utils import get_transform
from .random_noise import label_noise, image_noise
from .datasets import CIFAR10, CIFAR100, Nexperia, Nexperia_eval

from myImageFolder import MyImageFolder
from concatDataset import ConcatDataset

from torchvision import transforms

import os


def get_loader(args, data_aug=True):
    tform_train = get_transform(args, train=True, data_aug=data_aug)
    tform_test = get_transform(args, train=False, data_aug=data_aug)

    if args.dataset == 'cifar10':
        clean_train_set = CIFAR10(root=args.data_root, train=True, download=True, transform=tform_train)
        test_set = CIFAR10(root=args.data_root, train=False, download=True, transform=tform_test)

    
    elif args.dataset == 'cifar100':
        clean_train_set = CIFAR100(root=args.data_root, train=True, download=True, transform=tform_train)
        test_set = CIFAR100(root=args.data_root, train=False, download=True, transform=tform_test)
        
    elif args.dataset == 'nexperia':        
        image_datasets = {x: MyImageFolder(os.path.join(args.data_root, x),
                                          tform_train)
                  for x in ['train', 'val', 'test']}
                
        combined_dataset = ConcatDataset([image_datasets[x] for x in ['train', 'val', 'test']])

        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
        
        return dataloader, 10, np.load('files/targets.npy')
    
    elif args.dataset=='nexperia_split':
        data_transforms = {
            'train': tform_train,
            'val': tform_test,
            'test': tform_test
        }
        
        image_datasets = {x: MyImageFolder(os.path.join(args.data_root, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
                
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test'], 10, np.load('files/targets.npy')
    
    elif args.dataset=='nexperia_train':
        data_transforms = {
            'train': tform_train,
            'val': tform_test,
            'test': tform_test
        }
        
        image_datasets = {'train': Nexperia(args.data_root, args.train_set, data_transforms['train']),
                          'val': Nexperia(args.data_root, args.val_set, data_transforms['val']),
                          'test': Nexperia(args.data_root, args.test_set, data_transforms['test'])}
                
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
        
        train_labels = pd.read_csv(os.path.join(args.data_root, args.train_set),
                                   header=None, squeeze=True).str.rsplit(' ', n=1, expand=True).values[:,1].astype(np.int) - 1
        val_labels = pd.read_csv(os.path.join(args.data_root, args.val_set),
                                 header=None, squeeze=True).str.rsplit(' ', n=1, expand=True).values[:,1].astype(np.int) - 1
        test_labels = pd.read_csv(os.path.join(args.data_root, args.test_set),
                                  header=None, squeeze=True).str.rsplit(' ', n=1, expand=True).values[:,1].astype(np.int) - 1
        labels = np.concatenate((train_labels, val_labels, test_labels))
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test'], len(image_datasets['train'].classes), train_labels, val_labels, test_labels, labels, image_datasets['train'].class_to_idx['Pass']
    
    elif args.dataset=='nexperia_eval':
        
        image_dataset = Nexperia_eval(args.data_root, args.val_set, tform_test)
                
        dataloader = torch.utils.data.DataLoader(image_dataset, collate_fn=collate_fn, batch_size=args.batch_size, shuffle=True, num_workers=4)
        
        if 'csv' in args.val_set:
            val_labels = pd.read_csv(
                os.path.join(args.data_root, args.val_set), squeeze=True)['label'].map(image_dataset.class_to_idx).values
        elif 'txt' in args.val_set:
            val_labels = pd.read_csv(os.path.join(args.data_root, args.val_set),
                                     header=None, squeeze=True).str.rsplit(' ', n=1, expand=True).values[:,1].astype(np.int) - 1
        else:
            raise KeyError("Val set {} is not supported.".format(args.val_set))
        
        return dataloader, len(image_dataset.classes), val_labels, image_dataset.class_to_idx['Pass']
    
    elif args.dataset=='nexperia_merge':
        data_transforms = {
            'train': tform_train,
            'val': tform_test,
            'test': tform_test
        }
        
        image_datasets_1 = {x: MyImageFolder(os.path.join(args.data_root_1, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
        image_datasets_2 = {x: MyImageFolder(os.path.join(args.data_root_2, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
        if args.data_root_3 is not None:
            image_datasets_3 = {x: MyImageFolder(args.data_root_3, data_transforms[x])
                                         for x in ['train', 'val', 'test']}
            image_datasets = {x: ConcatDataset((image_datasets_1[x], image_datasets_2[x], image_datasets_3[x]))
                              for x in ['train', 'val', 'test']}

            train_set = np.loadtxt(args.train_set, dtype=str)[:,0]
            val_set = np.loadtxt(args.val_set, dtype=str)[:,0]
            test_set = np.loadtxt(args.test_set, dtype=str)[:,0]
            
            train_indices = []
            val_indices = []
            test_indices = []
            for i in range(len(image_datasets['train'])):
                if image_datasets['train'][i][-1]!=2 or image_datasets['train'][i][0][-1][
                    len('/import/home/share/SourceData/DownSampled/'):] in train_set:
                    train_indices.append(i)
            for i in range(len(image_datasets['val'])):
                if image_datasets['val'][i][-1]!=2 or image_datasets['val'][i][0][-1][
                    len('/import/home/share/SourceData/DownSampled/'):] in val_set:
                    val_indices.append(i)
            for i in range(len(image_datasets['test'])):
                if image_datasets['test'][i][-1]!=2 or image_datasets['test'][i][0][-1][
                    len('/import/home/share/SourceData/DownSampled/'):] in test_set:
                    test_indices.append(i)

            samplers = {'train': torch.utils.data.SubsetRandomSampler(list(train_indices)),
                        'val': torch.utils.data.SubsetRandomSampler(list(val_indices)),
                        'test': torch.utils.data.SubsetRandomSampler(list(test_indices))}
            
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, sampler=samplers[x], num_workers=4)
                           for x in ['train', 'val', 'test']}
        else:
            image_datasets = {x: ConcatDataset((image_datasets_1[x], image_datasets_2[x]))
                     for x in ['train', 'val', 'test']}
            
            dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size, shuffle=True, num_workers=4)
                           for x in ['train', 'val', 'test']}
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test'], 11, image_datasets
    
    elif args.dataset=='nexperia_month':
        return
    
    else:
        raise ValueError("Dataset `{}` is not supported yet.".format(args.dataset))
    
    if args.noise_rate > 0:
        noisy_train_set = deepcopy(clean_train_set)
        '''corrupt the dataset'''
        if args.noise_type == 'corrupted_label':
            label_noise(noisy_train_set, args)
        elif args.noise_type in ['Gaussian', 'random_pixels', 'shuffled_pixels']:
            image_noise(noisy_train_set, args)
        else:
            raise ValueError("Noise type {} is not supported yet.".format(args.noise_type))
        train_set = noisy_train_set
    else:
        print("Using clean dataset.")
        train_set = clean_train_set
    
    num_train = int(len(train_set) * 0.9)
    train_idx = list(range(num_train))
    val_idx = list(range(num_train, len(train_set)))
    
    if args.train_sets == 'trainval':
        train_loader = torch.utils.data.DataLoader(
            train_set, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    elif args.train_sets == 'train':
        train_subset = torch.utils.data.Subset(train_set, train_idx)
        train_loader = torch.utils.data.DataLoader(
            train_subset, batch_size=args.batch_size, shuffle=True,
            num_workers=args.workers, pin_memory=True)
    else:
        raise KeyError("Train sets {} is not supported.".format(args.train_sets))
    
    # for validation, we need to disable the data augmentation
    clean_train_set_for_val = deepcopy(clean_train_set)
    clean_train_set_for_val.transform = tform_test
    if args.noise_rate > 0:
        noisy_train_set_for_val = deepcopy(noisy_train_set)
        noisy_train_set_for_val.transform = tform_test

    val_sets = []
    if 'clean_set' in args.val_sets:
        val_sets.append(clean_train_set_for_val)
    if 'noisy_set' in args.val_sets:
        val_sets.append(noisy_train_set_for_val)
    if 'test_set' in args.val_sets:
        val_sets.append(test_set)
    if 'clean_train' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(clean_train_set_for_val, train_idx))
    if 'noisy_train' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(noisy_train_set_for_val, train_idx))
    if 'clean_val' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(clean_train_set_for_val, val_idx))
    if 'noisy_val' in args.val_sets:
        val_sets.append(torch.utils.data.Subset(noisy_train_set_for_val, val_idx))
    
    

    val_loaders = [
        torch.utils.data.DataLoader(
            val_set, batch_size=args.batch_size, shuffle=False,
            num_workers=args.workers, pin_memory=True)
        for val_set in val_sets
    ]

    test_loader = torch.utils.data.DataLoader(
        test_set, batch_size=args.batch_size, shuffle=False,
        num_workers=args.workers, pin_memory=True)
        
    return train_loader, val_loaders, test_loader, train_set.num_classes, np.asarray(train_set.targets)


def collate_fn(batch):
    batch = list(filter(lambda x: x is not None, batch))
    return torch.utils.data.dataloader.default_collate(batch)