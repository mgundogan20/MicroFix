# Xiu Li
# modified from
# 08-May-2015, Behzad Tabibian
import numpy as np
import scipy.io as sio
import matplotlib.pyplot as plt
import matplotlib.cm as cm
import cv2
import os


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

def bmpToPsfZemax(grid_h, grid_w, k_sz_small, jpeg_folder_path, save_path):
    # Converts PSFs given in jpeg format to npz format
    # The jpeg images contain a grid of kernels of shape 32x32.
    # The grid is of size grid_h x grid_w
    # The output npz file contains a PSF grid of shape (grid_h, grid_w, k_sz_small, k_sz_small, 3)

    img = cv2.imread(jpeg_folder_path)
    k_sz = 32

    PSF = np.zeros((grid_h, grid_w, k_sz, k_sz, 3), img.dtype)
    PSF_small = np.zeros((grid_h, grid_w, k_sz_small, k_sz_small, 3), img.dtype)
    for h_ in range(grid_h):
        for w_ in range(grid_w):
            local_patch = img[32*h_:32*(h_+1),
                              32*w_:32*(w_+1)]
            PSF[h_, w_] = local_patch
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

    bmpToPsfZemax(16, 16, 25, "../data/edmund_67462/45/psf.bmp","../data/Edmund_PSF_45.npz")
    bmpToPsfZemax(16, 16, 25, "../data/edmund_67462/20/psf.bmp","../data/Edmund_PSF_20.npz")
    bmpToPsfZemax(16, 16, 25, "../data/triplet/triplet.bmp","../data/triplet.npz")
    bmpToPsfZemax(16, 16, 25, "../data/triplet/triplet_20.bmp","../data/triplet_20.npz")
    










