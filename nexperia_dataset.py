import os
import numpy as np
import torch
from torch.utils.data import Dataset
from PIL import Image

class Nexperia(torch.utils.data.Dataset):
    def __init__(self, csv_file, root_dir, transform=None):
        """
        Args:
            csv_file (string): Path to the data file.
            root_dir (string): Directory with all the images.
            transform (callable, optional): Optional transform to be applied
                on a sample.
        """
        self.data = pd.read_csv(csv_file)
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
                                self.data.iloc[idx, 0])
        image = io.imread(img_name)
        if self.transform:
            image = self.transform(image)
        target = self.landmarks_frame.iloc[idx, 1]

        return image, target, idx, self.data.iloc[idx, 0]
    
    def num_classes(self):
        return len(np.unique(self.data[1]))