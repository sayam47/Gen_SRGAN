import torch
from torch.optim.lr_scheduler import MultiplicativeLR,MultiStepLR
from model.VGGStyleDiscriminator128 import VGGStyleDiscriminator128
from model.RRDBNet import RRDBNet

def get_generator(args):
	if args.model == 'rrdb':
		generator = RRDBNet()
		optimizer_G = torch.optim.Adam(generator.parameters(), lr=args.lr , betas=args.betas)
		if args.multistep_lr:
			scheduler_G = MultiStepLR(optimizer_G, milestones=args.multistep_milestones, gamma=args.multistep_gamma)
		else:
			lr_lambda = lambda epoch: 1 
			scheduler_G = MultiplicativeLR(optimizer, lr_lambda)
		return generator, optimizer_G, scheduler_G
	raise NotImplementedError(str(args.model) +" is not implemented")

def get_discriminator(args):
	if args.discriminator == 'vgg128':
		discriminator = VGGStyleDiscriminator128()
		optimizer_D = torch.optim.Adam(discriminator.parameters(), lr=args.lr , betas=args.betas)
		return discriminator, optimizer_D
	raise NotImplementedError(str(args.model) +" is not implemented")
