import torch
import torch.optim
import torch.nn.functional as F
import cv2
import os.path
import time
import os
import glob
import numpy as np
from collections import OrderedDict
from datetime import datetime
from scipy.signal import convolve2d
from models.uabcnet import UABCNet as net
import matplotlib.pyplot as plt
import utils.utils_image as util
import utils.utils_deblur as util_deblur
import utils.utils_train as util_train
import utils.utils_psf as util_psf

def strip_prefix_if_present(state_dict, prefix):
	keys = sorted(state_dict.keys())
	#if not all(key.startswith(prefix) for key in keys):
	#    return state_dict
	stripped_state_dict = OrderedDict()
	for key, value in state_dict.items():
		if key.startswith(prefix):
			stripped_state_dict[key.replace(prefix, "")] = value
	return stripped_state_dict

def make_size_divisible(img,stride):
	w,h,_ = img.shape

	w_new = w//stride*stride
	h_new = h//stride*stride

	return img[:w_new,:h_new,:]

def main(images_H,images_L,PSF_grid,ab_path=None,model_path="./logs/models/finetuned.pth"):
	# ----------------------------------------
	# load kernels
	# ----------------------------------------
	PSF_grid = np.load('./data/triplet_full_32x32.npz')['PSF']
	disp_width=3264
	
	PSF_grid = PSF_grid.astype(np.float32)
	print('PSF grid shape: {}'.format(PSF_grid.shape))

	gx,gy = PSF_grid.shape[:2]
	for xx in range(gx):
		for yy in range(gy):
			PSF_grid[xx,yy] = PSF_grid[xx,yy]/np.sum(PSF_grid[xx,yy],axis=(0,1))

	# ----------------------------------------
	# load model
	# ----------------------------------------
	stage = 8
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=stage, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2, sf=1, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	loaded_state = torch.load(model_path).state_dict()
	model.load_state_dict(loaded_state, strict=True)

	model.eval()
	for _, v in model.named_parameters():
		v.requires_grad = False
	model = model.to(device)
	
	print('Model loaded from {}'.format(model_path))
	# ----------------------------------------
	# load lambd/mu parameters
	# ----------------------------------------
	if ab_path is None:
		ab_numpy = np.ones((gx,gy,stage*2,3)).astype(np.float32)*0.01
	else:
		ab_numpy = np.loadtxt(ab_path).astype(np.float32).reshape(gx,gy,stage*2,3)
	ab = torch.tensor(ab_numpy,device=device,requires_grad=False)
	print("Lambd/mu parameters for HQS set")

	# ----------------------------------------
	# load images
	# ----------------------------------------
	images_L.sort()
	images_H.sort()
	print('Processing {} images'.format(len(images_L)))

	# ----------------------------------------
	# inference
	# ----------------------------------------
	for img_id,img_name in enumerate(images_L):
		img_full = make_size_divisible(cv2.imread(img_name),16)
		img_high = make_size_divisible(cv2.imread(images_H[img_id]),16)

		num_patch = [8,8]
		W, H = img_full.shape[0]//gx, img_full.shape[1]//gy
	
		img_result = np.zeros_like(img_full)
		# img_input = np.zeros_like(img_full)

		print("Patch size: [{},{}]".format(W,H))
		print("Chunk size: [{},{}]".format(W*num_patch[0],H*num_patch[1]))
		for i in range(gx//num_patch[0]):
			for j in range(gy//num_patch[1]):
				gx_ = i * num_patch[0]
				gy_ = j * num_patch[1]
				print('Processing chunk [{}/{}]'.format(i*gx//num_patch[0]+j+1,gx*gy//num_patch[0]//num_patch[1]))
				
				t0 = time.time()
				patch_L = img_full[gx_*W:(gx_+num_patch[0])*W,gy_*H:(gy_+num_patch[1])*H,:]

				px_start = 0
				py_start = 0

				PSF_patch = PSF_grid[gx_:gx_+num_patch[0],gy_:gy_+num_patch[1]]
				
				p_W,p_H= patch_L.shape[:2]
				expand = max(PSF_grid.shape[2]//2,p_W//16)
				block_expand = expand
				patch_L_wrap = util_deblur.wrap_boundary_liu(patch_L,(p_W+block_expand*2,p_H+block_expand*2))
				#centralize
				patch_L_wrap = np.hstack((patch_L_wrap[:,-block_expand:,:],patch_L_wrap[:,:p_H+block_expand,:]))
				patch_L_wrap = np.vstack((patch_L_wrap[-block_expand:,:,:],patch_L_wrap[:p_W+block_expand,:,:]))
				x = util.uint2single(patch_L_wrap)
				x = util.single2tensor4(x)

				k_all = []
				for h_ in range(num_patch[1]):
					for w_ in range(num_patch[0]):
						k_all.append(util.single2tensor4(PSF_patch[w_,h_]))
				k = torch.cat(k_all,dim=0)

				[x,k] = [el.to(device) for el in [x,k]]

				ab_patch = F.softplus(ab[gx_:gx_+num_patch[0],gy_:gy_+num_patch[1]])
				cd = []
				for h_ in range(num_patch[1]):
					for w_ in range(num_patch[0]):
						cd.append(ab_patch[w_:w_+1,h_])
				cd = torch.cat(cd,dim=0)

				x_E = model.forward_patchwise_SR(x,k,cd,num_patch,[W,H],sf=1).detach()
				x_E = x_E[...,block_expand:block_expand+p_W,block_expand:block_expand+p_H]

				patch_E = util.tensor2uint(x_E)

				t1 = time.time()
				print("Took: {:.2f}s".format(t1-t0))
				xk = patch_E
				xk = xk.astype(np.uint8)

				img_result[gx_*W:(gx_+num_patch[0])*W,gy_*H:(gy_+num_patch[1])*H,:] = cv2.resize(xk,(H*num_patch[0],W*num_patch[1]),interpolation=cv2.INTER_NEAREST)
				
				if i==2 and j==2:
					patch_H = img_high[gx_*W:(gx_+num_patch[0])*W,gy_*H:(gy_+num_patch[1])*H,:]
					util_train.save_triplet(f'./images/zemax/recovered/{img_id}_corner.png',patch_H[-200:,-200:],patch_L[-200:,-200:],patch_E[-200:,-200:])
				

		img_result = cv2.resize(img_result, (disp_width,disp_width*W//H))
		img_full = cv2.resize(img_full, (disp_width,disp_width*W//H))
		img_high = cv2.resize(img_high, (disp_width,disp_width*W//H))

		util_train.save_triplet(f'./images/zemax/recovered/{img_id}.png',img_high,img_full,img_result)


if __name__=='__main__':
	images_H = glob.glob("./images/zemax/high/*.png")
	images_L = glob.glob("./images/zemax/simulated/*.png")

	main(
		images_H=images_H,
		images_L=images_L,
		PSF_grid='./data/triplet_full_32x32.npz',
		ab_path='./logs/models/ab_finetuned.txt',
		model_path='./logs/models/finetuned.pth')
	
	print("Completed!")




