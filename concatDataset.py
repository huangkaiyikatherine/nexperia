import torch
import pandas as pd
import os
import bisect

class ConcatDataset(torch.utils.data.ConcatDataset):
    def __init__(self, datasets):
        super(ConcatDataset, self).__init__(datasets)
        # obtain bounding boxes
        defect_details = pd.DataFrame(columns=['label', 'defect_1', 'defect_2', 'shape', 'x_1', 'y_1', 'x_2', 'y_2'])
        for defect in ['chipping', 'lead_defect', 'foreign_material', 'pocket_damage', 'lead_glue', 'marking_defect', 'scratch']:
            file_name = os.path.join('/home/kaiyihuang/nexperia/new_data/20191129_Labeled_Image', defect,
                                     'ImageLabel.csv')
            table = pd.read_csv(file_name, names=['label', 'defect_2', 'shape', 'x_1', 'y_1', 'x_2', 'y_2'])
            table['defect_1'] = defect
            defect_details = defect_details.append(table)
        defect_details = defect_details.set_index(['defect_1', 'label'])
        self.defect_details = defect_details
        self.classes = ['chipping', 'device_flip', 'empty_pocket', 'foreign_material', 'good',
                        'lead_defect', 'lead_glue', 'marking_defect', 'pocket_damage', 'scratch']
        self.class_to_idx = {'chipping': 0,
                             'device_flip': 1,
                             'empty_pocket': 2,
                             'foreign_material': 3,
                             'good': 4,
                             'lead_defect': 5,
                             'lead_glue': 6,
                             'marking_defect': 7,
                             'pocket_damage': 8,
                             'scratch': 9}
        
        self.image_index_id = pd.read_csv('/home/kaiyihuang/nexperia/image_id_index.csv', index_col=0).sort_index()

    def __getitem__(self, idx):
        if idx < 0:
            if -idx > len(self):
                raise ValueError("absolute value of index should not exceed dataset length")
            idx = len(self) + idx
        dataset_idx = bisect.bisect_right(self.cumulative_sizes, idx)
        if dataset_idx == 0:
            sample_idx = idx
        else:
            sample_idx = idx - self.cumulative_sizes[dataset_idx - 1]
        return self.datasets[dataset_idx][sample_idx], idx, dataset_idx