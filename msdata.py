import os
import cv2
import random
from torch.utils.data import Dataset


class MSDataset(Dataset):
    def __init__(self, transform):
        self.hr_path = '../ms_set/gt_set/'
        self.rgb_path = '../ms_set/rgb_set/'
        self.lr_path = '../ms_set/depth_set/'
        self.hr = os.listdir(self.hr_path)
        self.hr.sort()
        self.rgb = os.listdir(self.rgb_path)
        self.rgb.sort()
        self.lr = os.listdir(self.lr_path)
        self.lr.sort()
        self.transform = transform
        self.patch_size = 64

    def __getitem__(self, idx):
        hr_img = cv2.imread(self.hr_path + self.hr[idx])
        hr_img = cv2.cvtColor(hr_img, cv2.COLOR_BGR2YCrCb)
        hr_Y = hr_img[:, :, 0:1]
        rgb_img = cv2.imread(self.rgb_path + self.rgb[idx])
        rgb_img = cv2.cvtColor(rgb_img, cv2.COLOR_BGR2YCrCb)
        rgb_Y = rgb_img[:, :, 0:1]
        lr_img = cv2.imread(self.lr_path + self.lr[idx])
        lr_img = cv2.cvtColor(lr_img, cv2.COLOR_BGR2YCrCb)
        lr_Y = lr_img[:, :, 0:1]

        hr_y, rgb_y, lr_y = self.get_patch(hr_Y, rgb_Y, lr_Y)
        hr_patch = self.transform(hr_y)
        rgb_patch = self.transform(rgb_y)
        lr_patch = self.transform(lr_y)

        return hr_patch, rgb_patch, lr_patch

    def __len__(self):
        return len(self.hr)

    def get_patch(self, hr, rgb, lr):
        h, w = hr.shape[:2]

        start_h = random.randint(0, h - self.patch_size)
        start_w = random.randint(0, w - self.patch_size)

        hr_p = hr[start_h:start_h + self.patch_size, start_w:start_w + self.patch_size, :]
        rgb_p = rgb[start_h:start_h + self.patch_size, start_w:start_w + self.patch_size, :]
        lr_p = lr[start_h:start_h + self.patch_size, start_w:start_w + self.patch_size, :]
        return hr_p, rgb_p, lr_p

