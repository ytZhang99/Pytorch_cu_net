import torch
import numpy as np
from torch.utils.data import Dataset


class MSDataset(Dataset):
    def __init__(self):
        self.hr = np.load('ms_set/gt.npy')
        self.rgb = np.load('ms_set/rgb.npy')
        self.lr = np.load('ms_set/depth.npy')

    def __getitem__(self, idx):
        hr_patch = torch.from_numpy(self.hr[idx])
        hr_patch = hr_patch.float()
        rgb_patch = torch.from_numpy(self.rgb[idx])
        rgb_patch = rgb_patch.float()
        lr_patch = torch.from_numpy(self.lr[idx])
        lr_patch = lr_patch.float()
        return hr_patch, rgb_patch, lr_patch

    def __len__(self):
        length = self.hr.shape[0]
        return length



