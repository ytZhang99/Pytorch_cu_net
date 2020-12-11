import numpy as np
import math


def calc_psnr(img1, img2):
    mse = np.mean((img1/255. - img2/255.) ** 2)
    pixel_max = 1.
    return 20 * math.log10(pixel_max / math.sqrt(mse))
