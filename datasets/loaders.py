from __future__ import absolute_import

from copy import deepcopy
import torch
import numpy as np

from .utils import get_transform
from .random_noise import label_noise, image_noise
from .datasets import CIFAR10, CIFAR100

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
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2,0.2,0.2),
                transforms.RandomAffine(degrees=2, translate=(0.15,0.1),scale=(0.75,1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]),
            'test': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]),
        }
        
        image_datasets = {x: MyImageFolder(os.path.join(args.data_root, x),
                                          data_transform)
                  for x in ['train', 'val', 'test']}
                
        combined_dataset = ConcatDataset([image_datasets[x] for x in ['train', 'val', 'test']])

        dataloader = torch.utils.data.DataLoader(combined_dataset, batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
        
        return dataloader, 10, np.load('/home/kaiyihuang/nexperia/targets.npy')
    
    elif args.dataset =='nexperia_split':
        data_transforms = {
            'train': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.RandomRotation(10),
                transforms.ColorJitter(0.2,0.2,0.2),
                transforms.RandomAffine(degrees=2, translate=(0.15,0.1),scale=(0.75,1.05)),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]),
            'val': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]),
            'test': transforms.Compose([
                transforms.Resize(255),
                transforms.CenterCrop(224),
                transforms.ToTensor(),
                transforms.Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]),
            ]),
        }
        
        image_datasets = {x: MyImageFolder(os.path.join(args.data_root, x),
                                          data_transforms[x])
                  for x in ['train', 'val', 'test']}
                
        dataloaders = {x: torch.utils.data.DataLoader(image_datasets[x], batch_size=args.batch_size,
                                             shuffle=True, num_workers=4)
              for x in ['train', 'val', 'test']}
        
        return dataloaders['train'], dataloaders['val'], dataloaders['test'], 10, np.load('/home/kaiyihuang/nexperia/targets.npy')
    
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
