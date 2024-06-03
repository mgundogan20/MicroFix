import torch
import torch.optim
import torch.nn.functional as F
import cv2
import os.path
import time
import os
import glob
import numpy as np
import utils.utils_image as util
import utils.utils_psf as util_psf
import utils.utils_deblur as util_deblur
import models.gan as gan
from utils.image_pool import ImagePool
from models.uabcnet import UABCNet as net
import copy
import matplotlib.pyplot as plt
import utils.utils_psf as util_psf
import utils.utils_train as util_train

def main(trainingDataHigh, model_load_path="./logs/models/uabcnet_pre.pth", kernel_path='./data/Edmund_PSF_45.npz', N_maxiter=1000, logs_directory="./logs", ab_path=None,trainindDataLow=None):
	# Global config
	sf = 1	
	stage = 8
	n_batch = 5
	patch_size = [128,128]
	patch_num = [2,2]
	disp_width = 512
	print("Global configs set.")

	# Load kernel
	PSF_grid = np.load(kernel_path)['PSF']
	PSF_grid = PSF_grid.astype(np.float32)
	gx, gy = PSF_grid.shape[:2]
	
	for w_ in range(gx):
		for h_ in range(gy):
			PSF_grid[w_,h_] = PSF_grid[w_,h_]/np.sum(PSF_grid[w_,h_],axis=(0,1))
	

	# Load model
	device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
	model = net(n_iter=8, h_nc=64, in_nc=4, out_nc=3, nc=[64, 128, 256, 512],
					nb=2,sf=sf, act_mode="R", downsample_mode='strideconv', upsample_mode="convtranspose")
	model.load_state_dict(torch.load(model_load_path),strict=True)
	model.train()
	for _, v in model.named_parameters():
		v.requires_grad = True
	model = model.to(device)
	print("Model is loaded to:", device)


	#positional lambda, mu for HQS.
	if ab_path is not None:
		ab_buffer = np.loadtxt(ab_path).reshape((gx,gy,2*stage,3)).astype(np.float32)
	else:
		ab_buffer = np.zeros((gx,gy,2*stage,3))
		ab_buffer[:,:,::2,:]=0.01
		ab_buffer[:,:,1::2,:]=0.1
	ab = torch.tensor(ab_buffer,dtype=torch.float32,device=device,requires_grad=True)
	print("Lambd/mu parameters for HQS set")
	
	
	params = []
	params += [{"params":[ab],"lr":5e-4}]
	for key,value in model.named_parameters():
		params += [{"params":[value],"lr":1e-5}]

	optimizer = torch.optim.Adam(params,lr=1e-4,betas=(0.9,0.999))

	scheduler = torch.optim.lr_scheduler.StepLR(optimizer,step_size=100,gamma=0.9)
	print("Optimizer set")

	#3.load training data
	imgs_H = trainingDataHigh
	imgs_H.sort()
	print("Training data loaded")

	# Take 10% images aside for validation
	imgs_val = imgs_H[:len(imgs_H)//10]
	imgs_H = imgs_H[len(imgs_H)//10:]
	print("Training on", len(imgs_H), "samples...")
	print("Validating on", len(imgs_val), "samples...")

	if trainindDataLow is not None:
		imgs_L = trainindDataLow
		imgs_L.sort()
		print(f"{len(imgs_L)} low resolution training images loaded.")

		imgs_L_val = imgs_L[:len(imgs_L)//10]
		imgs_L = imgs_L[len(imgs_L)//10:]
	print("\n")

	avg_loss = 0
	losses = []
	val_ssims = []
	val_ssims_L = []
	val_psnrs = []
	val_psnrs_L = []

	best_ssim = 0
	best_model = copy.deepcopy(model.state_dict())
	best_ab = copy.deepcopy(ab)

	global_iter = 0
	for i in range(N_maxiter):
		global_iter += 1

		# Pick a random image.
		img_idx = np.random.randint(len(imgs_H))
		img_H = cv2.imread(imgs_H[img_idx])

		if trainindDataLow is None:
			patch_L,patch_H,patch_psf,patch_ab = util_train.draw_training_pair(img_H,PSF_grid,ab,sf,patch_num,patch_size)
		else:
			img_L = cv2.imread(imgs_L[img_idx])
			patch_size = img_L.shape[0]//PSF_grid.shape[0], img_L.shape[1]//PSF_grid.shape[1]
			patch_L,patch_H,patch_psf,patch_ab = util_train.draw_training_pair(img_H,PSF_grid,ab,sf,patch_num,patch_size, image_L=img_L)
			
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


		x_E_grad = torch.gradient(x_E, dim=[-1, -2])
		x_gt_grad = torch.gradient(x_gt, dim=[-1, -2])
		loss = 2*F.mse_loss(x_E_grad[0]+x_E_grad[1],x_gt_grad[0]+x_gt_grad[1])

		loss_l1 = F.l1_loss(x_E,x_gt)
		loss += (0.2*loss_l1)
		losses.append(loss.item())
		avg_loss += loss / n_batch

		if global_iter%n_batch==0:
			optimizer.zero_grad()
			avg_loss.backward()
			optimizer.step()
			avg_loss=0

		if global_iter%100 == 0:
			patch_L = cv2.resize(patch_L,dsize=None,fx=sf,fy=sf,interpolation=cv2.INTER_NEAREST)
			patch_E = util.tensor2uint((x_E))
			util_train.save_triplet(f'{logs_directory}/images/fine{global_iter:05d}.png', patch_H, patch_L, patch_E)
			
			val_ssim, val_psnr, val_ssim_L, val_psnr_L = util_train.validate(imgs_val, PSF_grid, ab, sf, patch_num, patch_size, model, device)
			val_ssims.append(val_ssim)
			val_ssims_L.append(val_ssim_L)
			val_psnrs.append(val_psnr)
			val_psnrs_L.append(val_psnr_L)

			# if val_ssim > best_ssim:
			# 	best_ssim = val_ssim
			# 	best_model = copy.deepcopy(model)
			# 	best_ab = copy.deepcopy(ab)
			
			print("Validation completed for iteration {:05d}".format(global_iter))


	torch.save(model.state_dict(), f"{logs_directory}/models/finetuned.pth")
	print(f"Saved the best model to {logs_directory}/models/finetuned.pth")
	ab_numpy = ab.detach().cpu().numpy().flatten()
	np.savetxt(f'{logs_directory}/models/ab_finetuned.txt',ab_numpy)
	print(f"Saved the best lambda/mu parameters to {logs_directory}/models/ab_finetuned.txt")

	print("Saving the loss curve to ./logs/")
	_, (ax1, ax2, ax3) = plt.subplots(3)
	ax1.set(ylabel="training losses")
	ax1.plot(losses, color="red")
	ax1.legend(["SR"], loc="upper right", fontsize="small")
	
	ax2.plot(np.linspace(0, len(losses)-1,len(val_ssims)),val_ssims)
	ax2.plot(np.linspace(0, len(losses)-1,len(val_ssims)),val_ssims_L)
	ax2.set(ylabel="validation SSIMs")

	ax3.plot(np.linspace(0, len(losses)-1,len(val_psnrs)),val_psnrs)
	ax3.plot(np.linspace(0, len(losses)-1,len(val_psnrs)),val_psnrs_L)
	ax3.legend(["output", "input"], loc="lower right", fontsize="small")
	ax3.set(ylabel="validation PSNRs")
	
	plt.savefig(f"{logs_directory}/finetuning.png")
	plt.show()

if __name__ == '__main__':
	t0 = time.time()
	print("Fine tuning the model...")

	dataset_H = glob.glob('./images/zemax/high/*.png',recursive=True)
	dataset_H.sort()
	dataset_L = glob.glob('./images/zemax/simulated/*.png',recursive=True)
	dataset_L.sort()

	main(
		trainingDataHigh=dataset_H,
		model_load_path="./logs/models/finetuned.pth",
		kernel_path='./data/triplet_full_32x32.npz',
		N_maxiter=2000,
		logs_directory="./logs",
		trainindDataLow=dataset_L
		)
	
	deltaT = time.time() - t0
	print(f"Fine tuning completed in {deltaT/60:.2f} minutes.")
