# Pytorch implementation for TPAMI-CU-Net
This repository is Pytorch implementation for 2020 TPAMI paper entitled "Deep Convolutional Neural Network for Multi-modal Image Restoration and Fusion". [Paper Download](https://drive.google.com/file/d/1Nt4VOWNb8LxEt2TXd9OI0nNsFQSeCFeT/view?usp=sharing)

For the official TensorFlow code, please jump to [Official TensorFlow Code](https://github.com/cindydeng1991/TPAMI-CU-Net).
***
## Environment Requirement
    pytorch >= 1.4.1  
    opencv-python
***
## Prepair training and testing dataset
### training dataset
1. Put Depth/RGB/Groundtruth images in 'depth_set/rgb_set/gt_set' in the directory 'train_set', respectively.  
2. Open img2npy.py. Modify the variable 'source_path' in line 14 to 'depth_set/', 'rgb_set/' or 'gt_set/' and the variable 'save_name' in line 16 to 'depth', 'rgb' or 'gt' correspondingly.
3. Run the following command to transfer images to .npy files. These .npy files are placed in 'train_set'.  
`python img2npy.py`  
### testing dataset  
Put Depth/RGB/Groudtruth images in 'test_depth/test_rgb/test_gt' in the directory 'test_set', respectively.
## Train
Run the following command to train CU-Net.  
`python main.py`
## Test
Run the following command to test CU-Net.  
`python test.py`