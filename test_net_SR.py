import torch
import torch.optim
import torch.nn.functional as F

import cv2
import os
import glob
import numpy as np
import utils.utils_image as util
import utils.utils_train as util_train
import utils.utils_psf as util_psf
from models.uabcnet import UABCNet as net

np.random.seed(15)

def main(dataset, model_path='./logs/uabcnet.pth', ab_path=None, N_maxiter=5, save_path='./logs/test', kernel_path='./data'):
	#0. global config
	#scale factor
	sf = 1	
	stage = 8
	patch_size = [250,250]
	patch_num = [3,3]

	# Load kernel
	PSF_grid = np.load(kernel_path)['PSF']
	PSF_grid = PSF_grid.astype(np.float32)
	gx, gy = PSF_grid.shape[:2]
	
	for w_ in range(gx):
		for h_ in range(gy):
			PSF_grid[w_,h_] = PSF_grid[w_,h_]/np.sum(PSF_grid[w_,h_],axis=(0,1))
	

	#2. local model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2,sf=sf, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	loaded_state_dict = torch.load(model_path)
	model.load_state_dict(loaded_state_dict,strict=True)
	model.eval()
	for _, v in model.named_parameters():
		v.requires_grad = False
	model = model.to(device)

	#positional lambda, mu for HQS, set as free trainable parameters here.
	if ab_path is not None:
		ab_buffer = np.loadtxt(ab_path).reshape((gx,gy,2*stage,3)).astype(np.float32)
	else:
		ab_buffer = np.ones((gx,gy,2*stage,3),dtype=np.float32)*0.1
	ab = torch.tensor(ab_buffer,device=device,requires_grad=False)
	print("Lambd/mu parameters for HQS set")

	#3.load training data
	imgs_H = dataset
	imgs_H.sort()

	global_iter = 0


	all_PSNR = []

	for i in range(N_maxiter):
		global_iter += 1
		#draw random image.
		img_idx = np.random.randint(len(imgs_H))
		img_H = cv2.imread(imgs_H[img_idx])
		patch_L,patch_H,patch_psf,patch_ab = util_train.draw_training_pair(img_H,PSF_grid,ab,sf,patch_num,patch_size)

		x = util.uint2single(patch_L)
		x = util.single2tensor4(x)
		x_gt = util.uint2single(patch_H)
		x_gt = util.single2tensor4(x_gt)

		k_local = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				k_local.append(util.single2tensor4(patch_psf[w_,h_]))
		k = torch.cat(k_local,dim=0)
		[x,x_gt,k] = [el.to(device) for el in [x,x_gt,k]]
		
		ab_patch = F.softplus(patch_ab)
		ab_patch_v = []
		for h_ in range(patch_num[1]):
			for w_ in range(patch_num[0]):
				ab_patch_v.append(ab_patch[w_:w_+1,h_])
		ab_patch_v = torch.cat(ab_patch_v,dim=0)

		x_E = model.forward_patchwise_SR(x,k,ab_patch_v,patch_num,[patch_size[0],patch_size[1]],sf)


		patch_L = cv2.resize(patch_L,dsize=None,fx=sf,fy=sf,interpolation=cv2.INTER_NEAREST)
		patch_E = util.tensor2uint((x_E))

		util_train.save_triplet('{}/result{:03d}.png'.format(save_path,i+1), patch_H, patch_L, patch_E)
		
		psnr = cv2.PSNR(patch_E,patch_H)
		all_PSNR.append(psnr)

		print(util.calculate_psnr(patch_E,patch_H))
		print(util.calculate_ssim(patch_E,patch_H), "\n")


	np.savetxt(f"{save_path}/psnr.txt",all_PSNR)
	print(f"Test results saved on {save_path}/psnr.txt")

if __name__ == '__main__':
	imgs_H = []
	# imgs_H.extend(glob.glob('./images/DIV2K_train/*.png',recursive=True))
	imgs_H.extend(glob.glob('./images/cell_data/*.jpeg',recursive=True))

	main(
		dataset=imgs_H,
		model_path='./logs/models/finetuned.pth',
		ab_path='./logs/models/ab_finetuned.txt',
		N_maxiter=10,
		save_path='./logs/test',
		kernel_path='./data/triplet_full_32x32.npz')
