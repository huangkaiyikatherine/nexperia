import os
import numpy as np
import pandas as pd
import torch
from torch.utils.data import Dataset
from torchvision import datasets
from PIL import Image
from skimage import io, transform


class CIFAR10(datasets.CIFAR10):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR10, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 10


class CIFAR100(datasets.CIFAR100):
    def __init__(self, root, train=True, transform=None, target_transform=None, download=False):
        super(CIFAR100, self).__init__(root, train=train, transform=transform,
                                     target_transform=target_transform, download=download)

        # unify the interface
        if not hasattr(self, 'data'):       # torch <= 0.4.1
            if self.train:
                self.data, self.targets = self.train_data, self.train_labels
            else:
                self.data, self.targets = self.test_data, self.test_labels

    def __getitem__(self, index):
        img, target = self.data[index], self.targets[index]

        # doing this so that it is consistent with all other datasets
        # to return a PIL Image
        img = Image.fromarray(img)

        if self.transform is not None:
            img = self.transform(img)

        if self.target_transform is not None:
            target = self.target_transform(target)

        return img, target, index
    
    @property
    def num_classes(self):
        return 100
    

class Nexperia(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the data file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(os.path.join(root_dir, csv_file), header=None, squeeze=True).str.rsplit(' ', n=1, expand=True).values
        self.root_dir = root_dir
        self.transform = transform
        self.classes = ['Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass',
                        'Foreign_material', 'Empty_pocket', 'Device_flip', 'Chipping']
        self.class_to_idx = {'Others': 0,
                             'Marking_defect': 1,
                             'Lead_glue': 2,
                             'Lead_defect': 3,
                             'Pass': 4,
                             'Foreign_material': 5,
                             'Empty_pocket': 6,
                             'Device_flip': 7,
                             'Chipping': 8}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        img_name = os.path.join(self.root_dir,
                                self.data[idx, 0])
        image = Image.open(img_name).convert('RGB')
        if self.transform:
            image = self.transform(image)
        target = int(self.data[idx, 1]) - 1

        return image, target, idx, self.data[idx, 0]
    
    def num_classes(self):
        return len(np.unique(self.data[1]))


class Nexperia_eval(Dataset):
    def __init__(self, root_dir, csv_file, transform=None):
        """
        Args:
            csv_file (string): Path to the data file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        if 'csv' in csv_file:
            self.data = pd.read_csv(os.path.join(root_dir, csv_file), squeeze=True)
        elif 'txt' in csv_file:
            self.data = pd.read_csv(os.path.join(root_dir, csv_file), header=None, squeeze=True).str.rsplit(' ', n=1, expand=True).values
        else:
            raise KeyError("Val set {} is not supported.".format(self.csv_file))
        self.root_dir = root_dir
        self.csv_file = csv_file
        self.transform = transform
        self.classes = ['Others', 'Marking_defect', 'Lead_glue', 'Lead_defect', 'Pass',
                        'Foreign_material', 'Empty_pocket', 'Device_flip', 'Chipping']
        self.class_to_idx = {'Others': 0,
                             'Marking_defect': 1,
                             'Lead_glue': 2,
                             'Lead_defect': 3,
                             'Pass': 4,
                             'Foreign_material': 5,
                             'Empty_pocket': 6,
                             'Device_flip': 7,
                             'Chipping': 8}
        
    def __len__(self):
        return len(self.data)

    def __getitem__(self, idx):
        if torch.is_tensor(idx):
            idx = idx.tolist()

        if 'csv' in self.csv_file:
            img_file = self.data.iloc[idx]['id']
            target = self.class_to_idx[self.data.iloc[idx]['label']]
        elif 'txt' in self.csv_file:
            img_file = self.data[idx, 0]
            target = int(self.data[idx, 1]) - 1
        else:
            raise KeyError("Val set {} is not supported.".format(self.csv_file))
        img_name = os.path.join(self.root_dir, img_file)
        try:
            image = Image.open(img_name).convert('RGB')
        except:
            return None
        if self.transform:
            image = self.transform(image)

        return image, target, idx, img_file
    
    def num_classes(self):
        return len(np.unique(self.data[1]))