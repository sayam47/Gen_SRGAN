import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM
from torch.autograd import Variable


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

def psnr_fn(sr , hr , max_value = 1.):
	mse = torch.mean((sr - hr)**2 , axis = (1,2,3))
	return 20 * torch.log10(max_value / torch.sqrt(mse))

def mse_fn(sr , hr):
	return torch.mean((sr-hr)**2 , axis = (1,2,3))	

ssim_fn = SSIM(data_range = 1.0, size_average= False, channel = 3)

def ssim_fn_val(sr , hr , max_value = 255):
	return SSIM(data_range = max_value, size_average= False, channel = 3)(sr,hr)

# def get_loss_fn(loss_type , args = None):
# 	if loss_type == 'L1':
# 		return nn.L1Loss()
# 	if loss_type == 'MSE':
# 		return nn.MSELoss()
# 	if loss_type == 'PSNR':
# 		return psnr.PSNR()
# 	if loss_type == 'SSIM':
# 		return SSIM(data_range=1.0, size_average = False, channel = 3)
# 	if loss_type == 'GAN':
# 		return GAN.GAN(args)
# 	if loss_type == 'HV':
# 		return hvloss.HVLoss()
# 	if loss_type == 'VGG':
# 		return VGG.VGG(args)
# 	if loss_type == 'HV_VGG':
# 		return HV_VGG.HV_VGG(args)
# 	raise NotImplementedError(str(loss_type) + " is not implemented")	

# class Loss():
# 	def __init__(self , args):
# 		self.args = args
# 		self.length = len(args.loss)
# 		self.weights = []
# 		self.loss_fn = []
# 		self.loss_type = []
# 		for weight,loss_type in self.args.loss:
# 			self.weights.append(weight)
# 			self.loss_fn.append(self._get_loss_fn(loss_type))
# 			self.loss_type.append(loss_type)

# 	def _get_loss_fn(self, loss_type):
# 		if loss_type == 'PSNR':
# 			return PSNR.PSNR()
# 		if loss_type == 'SSIM':
# 			return SSIM(data_range=1.0, size_average = False, channel = 3)
# 		if loss_type == 'GAN':
# 			return GAN.GAN(self.args)
# 		if loss_type == 'HV':
# 			return HV.HV(self.args)
# 		if loss_type == 'VGG':
# 			return VGG.VGG(self.args)
# 		if loss_type == 'HV_VGG':
# 			return HV_VGG.HV_VGG(self.args)
# 		raise NotImplementedError(str(loss_type) + " is not implemented")	

# 	def loss(self , sr , hr):
# 		loss_values = np.zeros(self.length)
# 		loss = 0
# 		for i in range(self.length):
# 			loss_values[i] = self.loss_fn[i](sr , hr)
# 			loss += self.weights[i] * loss_values[i]
# 		return loss , loss_values

