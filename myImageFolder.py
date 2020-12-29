from torchvision import datasets
import pandas as pd
import os

class MyImageFolder(datasets.ImageFolder):
    def __init__(self, root, transform=None, target_transform=None):
        super(MyImageFolder, self).__init__(root=root,
                                          transform=transform,
                                          target_transform=target_transform)
        # obtain bounding boxes
        defect_details = pd.DataFrame(columns=['label', 'defect_1', 'defect_2', 'shape', 'x_1', 'y_1', 'x_2', 'y_2'])
        for defect in self.classes:
            if defect in ['chipping', 'lead_defect', 'foreign_material', 'pocket_damge', 'lead_glue', 'marking_defect', 'scratch']:
                file_name = os.path.join('/home/kaiyihuang/nexperia/new_data/20191129_Labeled_Image', defect,
                                         'ImageLabel.csv')
                table = pd.read_csv(file_name, names=['label', 'defect_2', 'shape', 'x_1', 'y_1', 'x_2', 'y_2'])
                table['defect_1'] = defect
                defect_details = defect_details.append(table)
        defect_details = defect_details.set_index(['defect_1', 'label'])
        self.defect_details = defect_details

    def __getitem__(self, index):
        path, target = self.samples[index]
        
        sample = self.loader(path)
        if self.transform is not None:
            sample = self.transform(sample)
        if self.target_transform is not None:
            target = self.target_transform(target)

        class_name = self.classes[target]
        if 'train' in path:
            image_id = path[42+len(class_name):-4]
        elif 'val' in path:
            image_id = path[40+len(class_name):-4]
        elif 'test' in path:
            image_id = path[41+len(class_name):-4]
        else:
            image_id = path[88+len(class_name):-4]
        
        return sample, target, index, image_id