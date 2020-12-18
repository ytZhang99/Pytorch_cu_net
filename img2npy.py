import os
import cv2
import numpy as np


# input images should be read in as BGR
def extract_y(img_bgr):
    img_yuv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2YCrCb)
    img_y = img_yuv[:, :, 0:1]
    return img_y


# image path
source_path = 'gt_set/'
save_path = './'
save_name = 'gt'

# image information
image_names = os.listdir(source_path)
image_names.sort()
# num_images = len(image_names) // 2
num_images = 600
img_h = 512
img_w = 512

# details for cropping images
patch_height = 64
patch_width = 64
num_channels = 1
stride = 32
row_patches = img_h // (patch_height - stride) - 1
col_patches = row_patches
each_img_patches = row_patches * col_patches
num_patches = each_img_patches * num_images

save_matrix = np.zeros((num_patches, num_channels, patch_height, patch_width))
print('The original size of saving matrix for dataset {} is {}.'.format(source_path, save_matrix.shape))

# normalization parameters
val_range = 255.
val_mean = 0.5
val_std = 0.5

num = 0
for idx in range(num_images):
    img = cv2.imread(source_path + image_names[idx])
    img = extract_y(img)
    img = (img / val_range - val_mean) / val_std
    img = np.transpose(img, (2, 0, 1))
    for i in range(row_patches):
        for j in range(col_patches):
            patch = img[:, stride * i: patch_height + stride * i, stride * j: patch_width + stride * j]
            save_matrix[num] = patch
            num = num + 1
print('The saving matrix for dataset is saved as {}.npy, size : {}'.format(save_name, save_matrix.shape))
np.save(save_path + save_name, save_matrix)
