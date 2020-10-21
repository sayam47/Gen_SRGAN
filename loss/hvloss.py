import torch
from torch import nn as nn
from torch.nn import functional as F

class HVLoss(nn.Module):
	def __init__(self, slack = 1.1):
		super(HVLoss, self).__init__()
		self.slack = slack

	def forward(self , *args):
		max_loss = args[0]
		for i in range(1,len(args)):
			loss = args[i]
			max_loss = torch.max(max_loss , loss)
		eta = (self.slack * max_loss).detach()
		loss_hv = 0
		for loss in args:
			loss_hv -= torch.log(eta - loss)
		return loss_hv.mean()