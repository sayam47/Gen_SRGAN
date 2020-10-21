import torch
import torch.nn as nn
import torch.nn.functional as F

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

class SSIMLoss(nn.Module):
	def __init__(self , max_value = 1.0):
		super(SSIMLoss, self).__init__()
		self.max_value = max_value
		self.ssim_fn = SSIM(data_range=max_value, size_average = False, channel = 3)

	def forward(self , sr , hr):
		ssim_val = self.ssim_fn(sr , hr)
		return 1 - ssim_val.mean()