import torch
import torch.nn as nn
import torch.nn.functional as F

class PSNR(nn.Module):
	def __init__(self , max_value = 1.0):
		self.max_value = max_value
		
	def forward(self , sr , hr):
		mse = torch.mean((sr - hr)**2 , axis = (1,2,3))
		return 20 * torch.log10(self.max_value / torch.sqrt(mse))
