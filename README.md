# MicroFix
Adapting the UNet architecture for image deconvolution to microscopy. 

The architecture used here was borrowed from https://arxiv.org/abs/2104.03078

## PSF Grids

1) Each PSF grid should be stored in a single .npz file.

2) A PSF grid is a 5D numpy array of shape:
	
		(grid_height, grid_width, kernel_height, kernel_width,channel count=3)
These files were generated using the Image Simulation Analysis feature of Zemax. There is a function which would crop the actual kernels from the padded output of the software.
After acquiring the PSFs, you can display them using
		python show_kernels.py
Which should then display PSF grids such as:

![image](https://github.com/mgundogan20/MicroFix/assets/72755125/884b2067-07b9-4f08-9e3d-b6cd4b1cdb1e)
![image](https://github.com/mgundogan20/MicroFix/assets/72755125/2c8eedaf-836e-4ed2-9a02-6ed6f19fdcbc)
![image](https://github.com/mgundogan20/MicroFix/assets/72755125/54698646-39df-425f-879b-8d131713d042)
![image](https://github.com/mgundogan20/MicroFix/assets/72755125/fce45475-8f33-49a9-9c65-94fe3c5af67c)


## Training Images
During training, a mixture of high resolution images from DIV2K dataset and our own microscopical images were used. We found out that natural images from DIV2K presents a variety within the training data, which allows more robust pretraining results.
![image](https://github.com/mgundogan20/MicroFix/assets/72755125/37ea8acb-7373-45ff-be87-c1269776ee96)
![image](https://github.com/mgundogan20/MicroFix/assets/72755125/04ddc47f-1ba2-4907-8432-cef2e1cb0700)

## Training
Training consists of two stages, during the pretraining phase various PSFs and image kinds are used to obtain a robust and flexible network.
In the second "finetuning" phase, the network is assigned a specific PSF, and is fed training pairs from that imaging system. To better recover the structural properties of the cells, we also utilize a modified loss function.

<img src="https://github.com/mgundogan20/MicroFix/assets/72755125/bb95ccfd-d786-407a-b878-b3c917e1395c" width="800px" alt="Ground Truth-Input-Output">

## Results
Down below you can see one of our results. The one in the left is the simulated output of our optics from Zemax. On the right you can see the output of the network.

![image](https://github.com/mgundogan20/MicroFix/assets/72755125/dc5f7c70-a9f2-4e34-bcb4-71f48743330a)

