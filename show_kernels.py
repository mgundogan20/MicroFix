import numpy as np
import cv2
import glob
from utils import utils_deblur as util_deblur

def gaussian_kernel_map(patch_num):
	# Generates a PSF_grid using util_deblur.gen_kernel() for each color RGB
	# Patch num is a tuple denoting the shape of the grid (GH x GW)
	# Returns PSF (GH x GW x H x W x C)
	PSF = np.zeros((patch_num[0],patch_num[1],25,25,3))
	for w_ in range(patch_num[0]):
		for h_ in range(patch_num[1]):
			PSF[w_,h_,...,0] = util_deblur.gen_kernel()
			PSF[w_,h_,...,1] = util_deblur.gen_kernel()
			PSF[w_,h_,...,2] = util_deblur.gen_kernel()
	return PSF

cv2.namedWindow("kernel", cv2.WINDOW_NORMAL)
cv2.resizeWindow("kernel", 600, 600)
grid = gaussian_kernel_map((10,12))

stacked_images = np.zeros((grid.shape[0]*grid.shape[2], grid.shape[1]*grid.shape[3], grid.shape[4]))

for row in range(grid.shape[0]):
	for col in range(grid.shape[1]):
		img = grid[row, col]
		stacked_images[row*grid.shape[2]:(row+1)*grid.shape[2], col*grid.shape[3]:(col+1)*grid.shape[3]] = img


cv2.imshow("kernel", stacked_images/stacked_images.max())
cv2.waitKey(0)


grids = glob.glob('./data/*.npz')
for path in grids:
	print(path)
	grid = np.load(path)['PSF']
	stacked_images = np.zeros((grid.shape[0]*grid.shape[2], grid.shape[1]*grid.shape[3], grid.shape[4]))

	for row in range(grid.shape[0]):
		for col in range(grid.shape[1]):
			img = grid[row, col]
			stacked_images[row*grid.shape[2]:(row+1)*grid.shape[2], col*grid.shape[3]:(col+1)*grid.shape[3]] = img
	cv2.imshow("kernel", stacked_images/stacked_images.max())
	cv2.waitKey(0)
	
cv2.destroyAllWindows()