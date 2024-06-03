# Xiu Li
# modified from
# 08-May-2015, Behzad Tabibian
import glob
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os

import utils.utils_deblur as util_deblur


def choose_psf(kernels,patch_num,psf_idx):
	psf = kernels[psf_idx]
	psf = psf[:patch_num[0],:patch_num[1]]
	return psf

def gaussian_kernel_map(patch_num):
	# Generates a PSF_grid using util_deblur.gen_kernel() for each color RGB
	# Patch num is a tuple denoting the shape of the grid (GH x GW)
	# Returns PSF (GH x GW x H x W x C)
	PSF = np.zeros((patch_num[0],patch_num[1],25,25,3))
	for w_ in range(patch_num[0]):
		for h_ in range(patch_num[1]):
			PSF[w_,h_,...,0] = util_deblur.gen_kernel(min_var=10)
			PSF[w_,h_,...,1] = util_deblur.gen_kernel(min_var=10)
			PSF[w_,h_,...,2] = util_deblur.gen_kernel(min_var=10)
	return PSF

def draw_random_kernel(kernels,patch_num):
	# Draws a random PSF_grid from kernels or generates if it's unavailable
	# Kernels is a list of N PSFs. N x (GH_i x GW_i x H_i x W_i x C_i)
	# Patch num is a tuple denoting the shape of the grid (GH x GW)
	# Returns PSF (GH x GW x H x W x C)
	n = len(kernels)
	i = np.random.randint(n+1)
	if n==i:
		psf = gaussian_kernel_map(patch_num)
	else:
		psf = kernels[i]
	return psf

def load_kernels(kernel_path):
	# Loads all PSF_grids (GH x GW x H x W x C) from .npz files in kernel_path
	# Normalizes them
	# Returns them in a list N x (GH_i x GW_i x H_i x W_i x C_i)
	kernels = []
	kernel_files = glob.glob(os.path.join(kernel_path,'*.npz'))
	kernel_files.sort()
	for kf in kernel_files:
		print("loaded", kf)
		PSF_grid = np.load(kf)['PSF']
		PSF_grid = normalize_PSF(PSF_grid)
		kernels.append(PSF_grid)
	return kernels

def normalize_PSF(psf, method='local'):
    psf = psf.astype(np.float32)
    gx, gy = psf.shape[:2]
    for yy in range(gy):
        for xx in range(gx):
            psf[xx, yy] = psf[xx, yy]/np.sum(psf[xx, yy], axis=(0, 1))
    return psf


def mv2pm(mv, K, mm2pixel):
    RY1 = np.array([[np.cos(mv[0]), 0, np.sin(mv[0])],
                    [0, 1, 0],
                    [-np.sin(mv[0]), 0, np.cos(mv[0])]])
    RX2 = np.array([[1.0, 0.0, 0.0],
                    [0, np.cos(mv[1]), -np.sin(mv[1])],
                    [0, np.sin(mv[1]), np.cos(mv[1])]])
    RZ3 = np.array([[np.cos(mv[2]), -np.sin(mv[2]), 0],
                    [np.sin(mv[2]), np.cos(mv[2]), 0],
                    [0, 0, 1]])
    R = RX2.dot(RY1).dot(RZ3)
    t = mv[3:]*mm2pixel
    P = K.dot(np.vstack((R.T, t)).T)
    return P

def bbox(arr):
    rows = np.any(arr, axis=1)
    cols = np.any(arr, axis=0)
    rmin, rmax = np.where(rows)[0][[0, -1]]
    cmin, cmax = np.where(cols)[0][[0, -1]]

    return rmin, rmax, cmin, cmax

def jpegToPsfHeidel(grid_h, grid_w, k_sz, k_sz_small, jpeg_folder_path, save_path):
    # Converts PSFs given in jpeg format to npz format
    # The jpeg images contain grid_h x grid_w kernels.
    # Each kernel of size k_size is padded with a single pixel white on all 4 sides.

    psf_img_r = cv2.imread(
        '{}/kernels_channels_1_nm_regularized.jpg'.format(jpeg_folder_path), cv2.IMREAD_GRAYSCALE)
    psf_img_g = cv2.imread(
        '{}/kernels_channels_3_nm_regularized.jpg'.format(jpeg_folder_path), cv2.IMREAD_GRAYSCALE)
    psf_img_b = cv2.imread(
        '{}/kernels_channels_4_nm_regularized.jpg'.format(jpeg_folder_path), cv2.IMREAD_GRAYSCALE)

    PSF = np.zeros((grid_h, grid_w, k_sz, k_sz, 3), psf_img_r.dtype)
    PSF_small = np.zeros(
        (grid_h, grid_w, k_sz_small, k_sz_small, 3), psf_img_r.dtype)

    for h_ in range(grid_h):
        for w_ in range(grid_w):
            PSF[h_, w_, :, :, 0] = psf_img_r[
                (k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,
                (k_sz+2)*w_+1:(k_sz+2)*(w_+1)-1]
            
            PSF[h_, w_, :, :, 1] = psf_img_g[
                (k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,
                1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1]
            
            PSF[h_, w_, :, :, 2] = psf_img_b[
                (k_sz+2)*h_+1:(k_sz+2)*(h_+1)-1,
                1+(k_sz+2)*w_:(k_sz+2)*(w_+1)-1]

            PSF_small[h_, w_] = cv2.resize(PSF[h_, w_], dsize=(k_sz_small, k_sz_small))

    PSF_all = PSF.astype(np.float32)
    PSF_all = np.sum(PSF_all, axis=(-1)).reshape(-1, k_sz, k_sz)
    all_bbox = [bbox(arr) for arr in PSF_all]
    all_bbox = np.array(all_bbox)
    np.savez(save_path, PSF=PSF_small)

def zemax2psf(grid_h, grid_w, k_sz_small, k_sz, pad_bot, pad_left, pad_top, pad_right, image_path, save_path):
    # Converts PSFs given in jpeg format to npz format
    # The jpeg images contain a grid of kernels of shape 32x32.
    # The grid is of size grid_h x grid_w
    # The output npz file contains a PSF grid of shape (grid_h, grid_w, k_sz_small, k_sz_small, 3)

    img = cv2.imread(jpeg_folder_path)
    
    PSF = np.zeros((grid_h, grid_w, k_sz, k_sz, 3), img.dtype)
    PSF_small = np.zeros((grid_h, grid_w, k_sz_small, k_sz_small, 3), img.dtype)

    for h_ in range(grid_h):
        for w_ in range(grid_w):
            x_start = w_*(k_sz+pad_left+pad_right) + pad_left
            y_start = h_*(k_sz+pad_top+pad_bot) + pad_top

            PSF[h_, w_] = img[y_start:y_start+k_sz, x_start:x_start+k_sz]
            PSF_small[h_, w_] = cv2.resize(PSF[h_, w_], dsize=(k_sz_small, k_sz_small))

    PSF_all = PSF.astype(np.float32)
    PSF_all = np.sum(PSF_all, axis=(-1)).reshape(-1, k_sz, k_sz)
    all_bbox = [bbox(arr) for arr in PSF_all]
    all_bbox = np.array(all_bbox)

    np.savez(save_path, PSF=PSF_small)

if __name__ == '__main__':
    # Creating npz files for the PSFs
    jpegToPsfHeidel(10, 12, 31, 25, "../data/psfs","../data/Bad_PSF_achromatic_small.npz")
    jpegToPsfHeidel(8, 12, 51, 25, "../data/achromatic_lens/psfs","../data/Heidel_PSF_achromatic_small.npz")
    jpegToPsfHeidel(7, 11, 49, 25, "../data/plano_convex_lens/psfs","../data/Heidel_PSF_plano_small.npz")

    zemax2psf(grid_h=16,
              grid_w=16, 
              k_sz_small=64, 
              k_sz=64, 
              pad_bot=18, 
              pad_left=18, 
              pad_top=20, 
              pad_right=20, 
              image_path="../data/triplet/1_sim.bmp", 
              save_path="../data/psf_trip_big.npz")
    
    zemax2psf(grid_h=32,
              grid_w=32,
              k_sz_small=64,
              k_sz=64,
              pad_bot=5,
              pad_left=18,
              pad_top=7,
              pad_right=20,
              image_path="../data/triplet/res_sim.bmp",
              save_path="../data/triplet_full_32x32.npz")










