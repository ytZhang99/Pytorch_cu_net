import os
import cv2
import time
import torch
import random
import matplotlib
import numpy as np
import torch.optim as optim
import matplotlib.pyplot as plt
from torch.nn import MSELoss
from msdata import MSDataset
from cu_net_MIR import CUNet
from calc_pnsr import calc_psnr
from torchvision import transforms
from torch.utils.data import DataLoader
from torchvision.utils import save_image


class Trainer:
    def __init__(self):
        self.epoch = 1000
        self.batch_size = 64
        self.criterion = MSELoss(reduction='mean')
        self.lr = 0.0001
        self.transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=[0.5], std=[0.5])])
        self.train_set = MSDataset()
        self.train_loader = DataLoader(self.train_set, batch_size=self.batch_size,
                                       shuffle=True, num_workers=0)
        self.model = CUNet().cuda()
        self.optimizer = optim.Adam(self.model.parameters(), lr=self.lr)
        self.scheduler = optim.lr_scheduler.StepLR(self.optimizer, step_size=30, gamma=0.9)
        self.train_loss = []
        self.val_psnr = []

    def train(self):
        seed = random.randint(1, 1000)
        print("===> Random Seed: [%d]" % seed)
        random.seed(seed)
        torch.manual_seed(seed)

        if os.path.exists('model/1000.pth'):
            print('===> Loading pre-trained model...')
            state = torch.load('model/1000.pth')
            self.train_loss = state['train_loss']
            self.model.load_state_dict(state['model'])
        for ep in range(1, self.epoch+1):
            self.model.train()
            epoch_loss = []
            for batch, (hr, rgb, lr) in enumerate(self.train_loader):
                hr = hr.cuda()
                rgb = rgb.cuda()
                lr = lr.cuda()

                self.optimizer.zero_grad()
                torch.cuda.synchronize()
                start_time = time.time()
                z = self.model(lr, rgb)
                loss = 1000 * self.criterion(z, hr)
                epoch_loss.append(loss.item())
                loss.backward()
                self.optimizer.step()
                torch.cuda.synchronize()
                end_time = time.time()
                if batch % 50 == 0 and batch != 0:
                    print('Epoch:{}\tcur/all:{}/{}\tAvg Loss:{:.4f}\tTime:{:.2f}'.format(ep, batch, len(self.train_loader), loss.item(), end_time-start_time))

            self.scheduler.step()
            self.train_loss.append(np.mean(epoch_loss))

            state = {
                'model': self.model.state_dict(),
                'train_loss': self.train_loss
            }
            torch.save(state, os.path.join('model', 'latest.pth'))
            if ep % 5 == 0:
                torch.save(state, os.path.join('model', str(ep)+'.pth'))
            matplotlib.use('Agg')
            fig1 = plt.figure()
            plot_loss_list = self.train_loss
            plt.plot(plot_loss_list)
            plt.savefig('train_loss_curve.png')


            val_psnr = self.test()
            print(val_psnr)
            self.val_psnr.append(val_psnr)
            fig2 = plt.figure()
            plt.plot(self.val_psnr)
            plt.savefig('val_curve.png')

            plt.close('all')

        print('===> Finished Training!')

    def test(self):
        lr_dir = '../ms_set/test_depth/'
        rgb_dir = '../ms_set/test_rgb/'
        gt_dir = '../ms_set/test_gt/'
        lr_img = os.listdir(lr_dir)
        lr_img.sort()
        rgb_img = os.listdir(rgb_dir)
        rgb_img.sort()
        gt_img = os.listdir(gt_dir)
        gt_img.sort()
        psnr_list = []
        save_path = 'test_result/'
        num_img = len(gt_img)

        state = torch.load('model/latest.pth')
        self.model.load_state_dict(state['model'])
        self.model.eval()

        with torch.no_grad():
            for idx in range(num_img):
                print(lr_img[idx])
                lr = cv2.imread(lr_dir + lr_img[idx])
                lr = cv2.cvtColor(lr, cv2.COLOR_BGR2YCrCb)
                lr_y = lr[:, :, 0:1]
                lr_t = self.transform(lr_y)
                lr_t = lr_t.unsqueeze(0)

                rgb = cv2.imread(rgb_dir + rgb_img[idx])
                rgb = cv2.cvtColor(rgb, cv2.COLOR_BGR2YCrCb)
                rgb_y = rgb[:, :, 0:1]
                rgb_t = self.transform(rgb_y)
                rgb_t = rgb_t.unsqueeze(0)

                gt = cv2.imread(gt_dir + gt_img[idx])
                gt = cv2.cvtColor(gt, cv2.COLOR_BGR2YCrCb)
                gt_y = gt[:, :, 0:1]

                lr_t = lr_t.cuda()
                rgb_t = rgb_t.cuda()
                sr = self.model(lr_t, rgb_t)

                output = sr.squeeze(0)
                output = (output + 1) * 127.5
                output = output.cpu().numpy()
                output = np.transpose(output, (1, 2, 0))
                output = output.astype(np.uint8)
                psnr = calc_psnr(output, gt_y)
                psnr_list.append(psnr)
                output = output[:, :, 0]
                cv2.imwrite(save_path + lr_img[idx], output)
        return np.mean(psnr_list)
