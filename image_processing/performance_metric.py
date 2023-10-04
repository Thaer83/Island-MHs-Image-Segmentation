from math import log10, sqrt
from sewar.full_ref import mse, rmse, psnr, uqi, ssim, ergas, scc, rase, sam, msssim, vifp
#from skimage.metrics import structural_similarity as ssim
#from skimage.metrics import peak_signal_noise_ratio as psnr
#from skimage.metrics import variation_of_information as vinf

import numpy as np
import math

"""  not correct (provide higher PSNR than other libraries
def My_PSNR(original, compressed):
    mse = np.mean((original - compressed) ** 2)
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        return 100
    max_pixel = 255.0
    psnr = 20 * log10(max_pixel / sqrt(mse))
    return psnr
"""
def My_PSNR(original, compressed):
    original = original.astype(np.float)
    compressed = compressed.astype(np.float)
    #M = original.size[0]   
    #N = original.size[1]
    error = original - compressed
    mse = sum(sum(error * error)) / (original.size);
    if(mse == 0):  # MSE is zero means no noise is present in the signal .
                  # Therefore PSNR have no importance.
        psnr = 99
    else:
        psnr = 10*math.log(255.0*255.0/mse) / math.log(10);
    
    return psnr


def Quality_Assessment(original, compressed):
    
    # peak signal to noise ratio
    psnr_v = psnr(original, compressed)
    
    # structural similarity index
    ssim_v = ssim(original, compressed)
    
    # Universal Quality Image Index
    uqi_v = uqi(original, compressed)
    
    # root mean squared error
    rmse_v = rmse(original, compressed)
    
    # spatial correlation coefficient 
    scc_v = scc(original, compressed)
    
    # Pixel Based Visual Information Fidelity
    vifp_v = vifp (original, compressed)
    return psnr_v, ssim_v, uqi_v, rmse_v, scc_v, vifp_v 
    
