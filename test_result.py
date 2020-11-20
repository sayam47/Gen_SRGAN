import argparse

import torch.nn as nn
import torch.nn.functional as F
import torch

from torch.utils.data import DataLoader
import torchvision.transforms as transforms

from torch.utils.data import Dataset
from torch.autograd import Variable

import os
import glob
import csv
from PIL import Image
from tqdm import tqdm

import numpy as np

from loss import psnr_fn, mse_fn, ssim_fn, ssim_fn_val
import model


parser = argparse.ArgumentParser(description='Test Super Resolution')

parser.add_argument('--model', type=str , default='rrdb' , help = 'architecture of generator')
parser.add_argument('--debug' , type=int , default=0 ,help='turn debug on')
parser.add_argument('--scale' , type=int , default=4 ,help='scale')
parser.add_argument('--lr' , type=float , default = 0.0001 , help = 'learning rate')
parser.add_argument('--betas' , type=str , default = "0.9,0.99" , help = 'betas for adam')
parser.add_argument('--multistep_lr' , type=int , default = 0, help = 'use multistep lr scheduler')
parser.add_argument('--multistep_milestones' , type=str , default = "5000,100000,200000,300000", help = 'milestones for multistep lr scheduler')
parser.add_argument('--multistep_gamma' , type=float , default = 0.5, help = 'gamma for multistep lr scheduler')

parser.add_argument('--test_data_path', type=str , default='/scratch/sayam.choudhary.cse17.iitbhu/SR_testing_datasets/Set5/' , help = 'location of input images')
parser.add_argument('--output_path' , type=str , default='/home/sayam.choudhary.cse17.iitbhu/test_output' , help = 'location of test folder')
parser.add_argument('--dataset_name' , type=str , default='Set5' , help = 'name of test dataset')
parser.add_argument('--model_path' , type=str , default='/scratch/sayam.choudhary.cse17.iitbhu/output_M4_l1_esrgan/saved_model/checkpoint_2100.pth' , help = 'path of saved model')
parser.add_argument('--model_name' , type=str , default='M4_l1' , help = 'name of model')
parser.add_argument('--batch_size' , type=int , default=1 , help = 'batch size of dataloader')

args = parser.parse_args()

args.debug = bool(args.debug)
args.multistep_lr = bool(args.multistep_lr)
args.multistep_milestones = list(map(int , args.multistep_milestones.split(',')))
args.betas = tuple(map(float , args.betas.split(',')))

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

# Normalization parameters for pre-trained PyTorch models
mean = np.array([0.485, 0.456, 0.406])
std = np.array([0.229, 0.224, 0.225])

class TestDatasetResult(Dataset):
	def __init__(self, args):
		# self.hr_shape = args.output_size
		# self.lr_shape = (self.hr_shape[0]//args.scale , self.hr_shape[1]//args.scale)
		self.files = sorted(glob.glob(args.test_data_path + "/*.*"))
		self.img_transform = transforms.Compose(
			[
				transforms.ToTensor(),
			]
		)
		if args.debug:
			print(len(self.files) , "Testing Images of " , args.dataset_name)

	def __getitem__(self , index):
		img_hr = Image.open(self.files[index % len(self.files)]).convert('RGB')
		hr_shape = list(img_hr.size)
		if hr_shape[0]%args.scale != 0 or hr_shape[1]%args.scale != 0:
			right_pad = 0
			bottom_pad = 0
			if hr_shape[0]%args.scale != 0:
				right_pad = (args.scale - (hr_shape[0]%args.scale))
				hr_shape[0] += (args.scale - (hr_shape[0]%args.scale))
			if hr_shape[1]%args.scale != 0:
				bottom_pad = (args.scale - (hr_shape[1]%args.scale))
				hr_shape[1] += (args.scale - (hr_shape[1]%args.scale))
			img_hr = transforms.Pad((0 , 0, right_pad , bottom_pad), padding_mode='constant')(img_hr)
		lr_shape = (hr_shape[0]//args.scale , hr_shape[1]//args.scale)
		img_lr = transforms.Resize((lr_shape[1],lr_shape[0]) , Image.BICUBIC)(img_hr)
		img_lr = self.img_transform(img_lr)
		img_hr = self.img_transform(img_hr)
		img_lr = transforms.Normalize(mean, std)(img_lr)
		return {"lr": img_lr , "hr": img_hr}
	
	def __len__(self):
		return len(self.files)

def get_dataloader(args):
	dataset = TestDatasetResult(args)
	num_worker = 2
	shuffle = False
	batch_size = args.batch_size
	return DataLoader(
			dataset,
			batch_size=batch_size,
			shuffle=shuffle,
			num_workers=num_worker,
		)

def write_to_csv_file(filename , content):
	with open(filename , 'a+') as f:
		file_writer = csv.writer(f)
		file_writer.writerow(content)


@torch.no_grad()
def test(generator , dataloader):
	psnr_val = 0
	ssim_val = 0
	for i,imgs in tqdm(enumerate(dataloader)):
		imgs_hr = Variable(imgs['hr'].type(Tensor))	
		imgs_lr = Variable(imgs['lr'].type(Tensor))
		gen_hr =  generator(imgs_lr)

		# Scaling output
		gen_hr = gen_hr.clamp(0,1)
		
		psnr_val += psnr_fn(gen_hr , imgs_hr).mean().item()
		ssim_val += ssim_fn(gen_hr , imgs_hr).mean().item()
	psnr_val /= len(dataloader)
	ssim_val /= len(dataloader)
	write_to_csv_file( os.path.join(args.output_path ,'test_log.csv' ) , [args.dataset_name , args.model_name , args.model_path , psnr_val , ssim_val])


@torch.no_grad()
def get_generator():
	print("GPU : " , torch.cuda.is_available())	
	
	generator,optimizer_G,scheduler_G = model.get_generator(args)
	generator.to(device)

	if not torch.cuda.is_available():
		checkpoint = torch.load(args.model_path , map_location=torch.device('cpu'))
	else:
		checkpoint = torch.load(args.model_path)

	epoch = checkpoint['epoch']
	generator.load_state_dict(checkpoint['gen_state_dict'])

	return generator


if __name__ == '__main__':
	generator = get_generator()
	dataloader = get_dataloader(args)
	test(generator , dataloader)

