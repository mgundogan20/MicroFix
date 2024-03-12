import torch
import torch.optim
import torch.nn.functional as F

import cv2
import os.path
import time
import os
import glob
import numpy as np
import matplotlib.pyplot as plt
import utils.utils_image as util
import utils.utils_deblur as util_deblur
import utils.utils_psf as util_psf
from models.uabcnet import UABCNet as net

def load_kernels(kernel_path):
	# Loads all PSF_grids (GH x GW x H x W x C) from .npz files in kernel_path
	# Returns them in a list N x (GH_i x GW_i x H_i x W_i x C_i)
	kernels = []
	kernel_files = glob.glob(os.path.join(kernel_path,'*.npz'))
	kernel_files.sort()
	for kf in kernel_files:
		PSF_grid = np.load(kf)['PSF']
		PSF_grid = util_psf.normalize_PSF(PSF_grid)
		kernels.append(PSF_grid)
	return kernels

def draw_random_kernel(kernels,patch_num):
	# Draws a random PSF_grid from kernels or generates if it's unavailable
	# Kernels is a list of N PSFs. N x (GH_i x GW_i x H_i x W_i x C_i)
	# Patch num is a tuple denoting the shape of the grid (GH x GW)
	# Returns PSF (GH x GW x H x W x C)
	n = len(kernels)
	i = np.random.randint(2*n)
	if n>0:
		psf = kernels[i]
	else:
		psf = gaussian_kernel_map(patch_num)
	return psf

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


def draw_training_pair(image_H,psf,sf,patch_num,patch_size,image_L=None):
	# image_H is the ground truth
	# psf is a grid of PSFs (GH x GW x H x W x C)
	# image_L is the aberrated image. When not provided, it's generated by the function
	# Returns patch_H, patch_L, patch_PSF

	w,h = image_H.shape[:2]
	gx,gy = psf.shape[:2]
	px_start = np.random.randint(0,gx-patch_num[0]+1)
	py_start = np.random.randint(0,gy-patch_num[1]+1)

	# TODO: Add the option to focus on edges

	psf_patch = psf[px_start:px_start+patch_num[0],py_start:py_start+patch_num[1]]
	patch_size_H = [patch_size[0]*sf,patch_size[1]*sf]

	if image_L is None:
		#generate image_L on-the-fly
		conv_expand = psf.shape[2]//2
		x_start = np.random.randint(0,w-patch_size_H[0]*patch_num[0]-conv_expand*2+1)
		y_start = np.random.randint(0,h-patch_size_H[1]*patch_num[1]-conv_expand*2+1)
		patch_H = image_H[x_start:x_start+patch_size_H[0]*patch_num[0]+conv_expand*2,\
		y_start:y_start+patch_size_H[1]*patch_num[1]+conv_expand*2]
		patch_L = util_deblur.blockConv2d(patch_H,psf_patch,conv_expand)

		patch_H = patch_H[conv_expand:-conv_expand,conv_expand:-conv_expand]
		patch_L = patch_L[::sf,::sf]

		#wrap_edges around patch_L to avoid FFT boundary effect.
		#wrap_expand = patch_size[0]//8
		# patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(patch_size[0]*patch_num[0]+wrap_expand*2,\
		# patch_size[1]*patch_num[1]+wrap_expand*2))
		# patch_L_wrap = np.hstack((patch_L_wrap[:,-wrap_expand:,:],patch_L_wrap[:,:patch_size[1]*patch_num[1]+wrap_expand,:]))
		# patch_L_wrap = np.vstack((patch_L_wrap[-wrap_expand:,:,:],patch_L_wrap[:patch_size[0]*patch_num[0]+wrap_expand,:,:]))
		# patch_L = patch_L_wrap

	else:
		x_start = px_start * patch_size_H[0]
		y_start = py_start * patch_size_H[1]
		x_end = x_start + (patch_size_H[0]*patch_num[0])
		y_end = y_start + (patch_size_H[1]*patch_num[1])
		patch_H = image_H[x_start:x_end, y_start:y_end]

		x_start = px_start * patch_size[0]
		y_start = py_start * patch_size[1]
		x_end = x_start + (patch_size[0]*patch_num[0])
		y_end = y_start + (patch_size[1]*patch_num[1])
		patch_L = image_L[x_start:x_end, y_start:y_end]

	return patch_L,patch_H,psf_patch

def main():
	#0. global config
	#scale factor
	sf = 4					# Scaling factor between Resunet Layers
	stage = 8
	patch_size = [32,32]
	patch_num = [3,3]

	#1. local PSF
	# Takes .npz files which can be generated by utils_psf.py
	# Normalizes them
	# Returns them in a list
	# Each psf is of the form (grid_height, grid_width, kernel_size, kernel_size, channel count)
	
	#shape: gx,gy,kw,kw,3
	all_PSFs = load_kernels('./data')


	#2. local model
	# Defines the architecture, the one used by Li et. al. can be described as follows
	# model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
	# 				nb=2,sf=sf, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	# Head	x-x1		(conv from 4 to 64 channels)
	# Down1	x1-x2		(2*res 64 to 64 to 64 + down 64-128)
	# Down2	x2-x3		(2*res 128 to 128 to 128 + down 128 to 256)
	# Down3	x3-x4		(2*res 256 to 256 to 256 + down 256 to 512)
	# Body	x4-x		(2*res 512 to 512 to 512)
	# Up3	x+x4-x		(up 512 to 256 + 2* res 256 to 256 to 256)
	# Up2	x+x3-x		(up 256 to 128 + 2*res 128 to 128 to 128)
	# Up1	x+x2-x		(up 128 to 64 + 2*res 64 to 64 to 64)
	# Tail	x+x1-x		(conv from 64 to 3)

	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2,sf=sf, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	#model.proj.load_state_dict(torch.load('./data/usrnet_pretrain.pth'),strict=True)
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)

	#positional lambda, mu for HQS, set as free trainable parameters here.
	ab_buffer = np.ones((patch_num[0],patch_num[1],2*stage,3),dtype=np.float32)*0.1
	ab = torch.tensor(ab_buffer,device=device,requires_grad=True)

	params = []
	params += [{"params":[ab],"lr":0.0005}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":0.0001}]
	optimizer = torch.optim.Adam(params,lr=0.0001,betas=(0.9,0.999))
	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=1000,gamma=0.9)

	#3.load training data
	imgs_H = glob.glob('./images/DIV2K_train/*.png',recursive=True)
	imgs_H.sort()

	# The following can be uncommented to provide aberrated images directly
	# But the draw_training_pair() call within main() should also be modified accordingly
	# imgs_L = glob.glob('./images/DIV2K_lr/*.png', recursive=True)
	# imgs_L.sort()

	global_iter = 0
	N_maxiter = 200000


	for i in range(N_maxiter):

		t0 = time.time()
		#draw random image.
		img_idx = np.random.randint(len(imgs_H))
		img_H = cv2.imread(imgs_H[img_idx])

		#draw random kernel
		PSF_grid = draw_random_kernel(all_PSFs,patch_num)

		# Cuts a random patch from the original image and the psf
		# Creates the noisy version
		patch_L,patch_H,patch_psf = draw_training_pair(img_H,PSF_grid,sf,patch_num,patch_size)

		# Time to generate the data
		t_data = time.time()-t0

		# Converts both the original and noisy patches to tensor4 objects
		x = util.uint2single(patch_L)
		x = util.single2tensor4(x)
		x_gt = util.uint2single(patch_H)
		x_gt = util.single2tensor4(x_gt)

		k_local = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				k_local.append(util.single2tensor4(patch_psf[w_,h_]))
		k = torch.cat(k_local,dim=0)

		# Data are moved to the gpu
		[x,x_gt,k] = [el.to(device) for el in [x,x_gt,k]]
		
		ab_patch = F.softplus(ab)
		ab_patch_v = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				ab_patch_v.append(ab_patch[w_:w_+1,h_])
		ab_patch_v = torch.cat(ab_patch_v,dim=0)

		# One forward pass is calculated
		x_E = model.forward_patchwise_SR(x,k,ab_patch_v,patch_num,[patch_size[0],patch_size[1]],sf)

		# Corresponding loss and gradiants are calculated
		# Weights are updated
		loss = F.l1_loss(x_E,x_gt)
		optimizer.zero_grad()
		loss.backward()
		optimizer.step()
		scheduler.step()

		# Time to update model parameters after data preprocessing
		t_iter = time.time() - t0 - t_data

		print('[iter:{}] loss:{:.4f}, data_time:{:.2f}s, net_time:{:.2f}s'.format(global_iter+1,loss.item(),t_data,t_iter))

		# Display ground_truth (patch_H), aberrated image(patch_L) and recovered image(patch_E)
		patch_L = cv2.resize(patch_L,dsize=None,fx=sf,fy=sf,interpolation=cv2.INTER_NEAREST)
		patch_E = util.tensor2uint((x_E))
		show = np.hstack((patch_H,patch_L,patch_E))
		cv2.imshow('H,L,E',show)
		key = cv2.waitKey(1)
		global_iter+= 1

		# for logging model weight.
		# if global_iter % 100 ==0:
		# 	torch.save(model.state_dict(),'./logs/uabcnet_{}.pth'.format(global_iter))

		if key==ord('q'):
			break
		if key==ord('s'):
			torch.save(model.state_dict(),'./logs/uabcnet.pth')

	torch.save(model.state_dict(),'./logs/uabcnet.pth')

if __name__ == '__main__':

	main()
