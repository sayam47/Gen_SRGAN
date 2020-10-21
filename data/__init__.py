import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.utils.data import Dataset

import glob
from PIL import Image

import numpy as np

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

def denormalize(input_tensor):
	if input_tensor.ndim == 3:
		input_tensor = input_tensor.unsqueeze(0)
	for tensor in input_tensor:
		for t, m, s in zip(tensor, mean, std):
			t.mul_(s).add_(m)
	return input_tensor

class TrainDataset(Dataset):
	def __init__(self, args):
		self.hr_shape = args.patch_size
		self.lr_shape = (self.hr_shape[0]//args.scale , self.hr_shape[1]//args.scale)
		self.files = sorted(glob.glob(args.train_data_path + "/*.*"))
		self.img_transform = transforms.Compose(
			[
				transforms.ToTensor(),
			]
		)	

		self.hr_transform = transforms.Compose(
			[
				transforms.RandomCrop(self.hr_shape , pad_if_needed=True, padding_mode='constant'),
				transforms.RandomHorizontalFlip(p=0.5),
				transforms.RandomVerticalFlip(p=0.5),
				transforms.RandomRotation((90,90)),
			]
		)
		if args.debug:
			print(len(self.files) , "Training Images")

	def __getitem__(self , index):
		img = Image.open(self.files[index % len(self.files)])
		img_hr = self.hr_transform(img)
		img_lr = transforms.Resize(self.lr_shape , Image.BICUBIC)(img_hr)
		img_hr = self.img_transform(img_hr)
		img_lr = self.img_transform(img_lr)
		img_lr = transforms.Normalize(mean, std)(img_lr)
		return {"lr": img_lr , "hr": img_hr}

	def __len__(self):
		return len(self.files)

class TestDataset(Dataset):
	def __init__(self, args):
		self.hr_shape = args.output_size
		self.lr_shape = (self.hr_shape[0]//args.scale , self.hr_shape[1]//args.scale)
		self.files = sorted(glob.glob(args.test_data_path + "/*.*"))
		self.img_transform = transforms.Compose(
			[
				transforms.ToTensor(),
			]
		)	

		self.hr_transform = transforms.Compose(
			[
				transforms.RandomCrop(self.hr_shape , pad_if_needed=True, padding_mode='constant'),
			]
		)
		if args.debug:
			print(len(self.files) , "Testing Images")

	def __getitem__(self , index):
		img = Image.open(self.files[index % len(self.files)])
		img_hr = self.hr_transform(img)
		img_lr = transforms.Resize(self.lr_shape , Image.BICUBIC)(img_hr)
		img_hr = self.img_transform(img_hr)
		img_lr = self.img_transform(img_lr)
		img_lr = transforms.Normalize(mean, std)(img_lr)
		return {"lr": img_lr , "hr": img_hr}
	
	def __len__(self):
		return len(self.files)

	
def dataloader(args , train = True):
	dataset = ""
	if train:
		dataset = TrainDataset(args)
	else:
		dataset = TestDataset(args)
	num_worker = args.n_threads
	if not train:
		num_worker = 2
	shuffle = True
	if not train:
		shuffle = False
	batch_size = args.batch_size
	if not train:
		batch_size = 5
	return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=args.n_threads,
		)