from torch.utils.data import Dataset
import os
import torch
import numpy as np

class LatentDataset(Dataset):
    def __init__(self, data_dir, transform=None):
        self.transform = transform
        
        self.samples = []
        for root, _, files in os.walk(data_dir):
            for file in files:
                if file.endswith('.npz'):
                    self.samples.append(os.path.join(root, file))
        self.lenth = len(self.samples)

    def __len__(self):
        return self.lenth
    
    def __getitem__(self, idx):
        path = self.samples[idx]
        np_data = np.load(path)
        x = torch.from_numpy(np_data['x'])
        y = torch.from_numpy(np_data['y'])
        
        return x, y