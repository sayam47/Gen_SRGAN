from option import args

import loss
import model
import data

from loss.VGGFeatureExtractor import VGGFeatureExtractor
from loss.hvloss import HVLoss
from loss.ssim_loss import SSIMLoss

from loss import psnr_fn, mse_fn, ssim_fn, ssim_fn_val

from data import denormalize

from tqdm import tqdm
import numpy as np
import csv
import os
import matplotlib.pyplot as plt

import torch
from torch.autograd import Variable

from torchvision.utils import save_image
import torch.nn.functional as F

from pytorch_msssim import ssim, ms_ssim, SSIM, MS_SSIM

device = torch.device("cuda" if torch.cuda.is_available() else "cpu")	
Tensor = torch.cuda.FloatTensor if torch.cuda.is_available() else torch.Tensor

ssim_fn_vgg = SSIM(data_range = 1.0, size_average= False, channel = 512)

def write_to_csv_file(filename , content):
	with open(filename , 'a+') as f:
		file_writer = csv.writer(f)
		file_writer.writerow(content)

def get_image(input_tensor):
	return input_tensor.clamp(0,1).permute(1,2,0).cpu().numpy()

@torch.no_grad()
def test(epoch , generator , dataloader):
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
	write_to_csv_file( os.path.join(args.output_path ,'test_log.csv' ) , [epoch , psnr_val , ssim_val])
	
@torch.no_grad()
def plot_image(epoch , generator , dataloader , dim=(1, 3), figsize=(15, 5)):
	for i,imgs in tqdm(enumerate(dataloader)):
		imgs_hr = Variable(imgs["hr"].type(Tensor))
		imgs_lr = Variable(imgs["lr"].type(Tensor))
		gen_hr = generator(imgs_lr)

		#denormalize input
		imgs_lr = denormalize(imgs_lr)

		# Scaling output
		gen_hr = gen_hr.clamp(0,1)

		for j in range(imgs_lr.shape[0]):
			batches_done = i * len(dataloader) + j
			psnr_val = psnr_fn(gen_hr[[j]] , imgs_hr[[j]]).mean().item()
			ssim_val = ssim_fn(gen_hr[[j]] , imgs_hr[[j]]).mean().item()

			file_name = os.path.join(args.output_path ,'generated_image_' + str(batches_done) + "_" + str(epoch) + "_PSNR : " + str(round(psnr_val ,2)) + " SSIM : " + str(round(ssim_val , 2)) + '.png')

			lr_image = F.interpolate(imgs_lr[[j]] , (imgs_hr.shape[2] , imgs_hr.shape[3]) , mode='nearest')[0]
			hr_image = imgs_hr[j]
			gen_image = gen_hr[j]

			concat_image = torch.cat((lr_image , gen_image , hr_image), 2)

			save_image(concat_image , file_name)


def normalize_VGG_features(input_tensor):
	assert(input_tensor.ndim == 4)
	maxi = input_tensor.max(2)[0].max(2)[0].unsqueeze(2).unsqueeze(3).detach()
	mini = input_tensor.min(2)[0].max(2)[0].unsqueeze(2).unsqueeze(3).detach()
	return (input_tensor-mini)/(maxi - mini)


def train():
	print("GPU : " , torch.cuda.is_available())	
	
	generator,optimizer_G,scheduler_G = model.get_generator(args)
	generator.to(device)

	discriminator,optimizer_D = model.get_discriminator(args)
	discriminator.to(device)

	start_epoch = 0

	if args.resume_training:
		if args.debug:
			print("Resuming Training")
		checkpoint = torch.load(args.checkpoint_path)
		start_epoch = checkpoint['epoch']
		generator.load_state_dict(checkpoint['gen_state_dict'])
		optimizer_G.load_state_dict(checkpoint['gen_optimizer_dict'])
		scheduler_G.load_state_dict(checkpoint['gen_scheduler_dict'])
		discriminator.load_state_dict(checkpoint['dis_state_dict'])
		optimizer_D.load_state_dict(checkpoint['dis_optimizer_dict'])

	feature_extractor = VGGFeatureExtractor().to(device)

	# Set feature extractor to inference mode
	feature_extractor.eval()

	# Losses
	bce_loss = torch.nn.BCEWithLogitsLoss().to(device)
	l1_loss = torch.nn.L1Loss().to(device)
	l2_loss = torch.nn.MSELoss().to(device)
	
	# equal to negative of hypervolume of input losses
	hv_loss = HVLoss().to(device)

	# 1 - ssim(sr , hr)
	ssim_loss = SSIMLoss().to(device)

	dataloader = data.dataloader(args)
	test_dataloader = data.dataloader(args , train = False)

	batch_count = len(dataloader)

	generator.train()

	loss_len = 6
	if args.method == 'M4' or args.method == 'M7':
		loss_len = 7
	elif args.method == 'M6':
		loss_len = 9

	losses_log = np.zeros(loss_len+1)

	for epoch in range(start_epoch , start_epoch + args.epochs):
		# print("*"*15 , "Epoch :" , epoch , "*"*15)
		losses_gen = np.zeros(loss_len)
		for i,imgs in tqdm(enumerate(dataloader)):
			batches_done = epoch * len(dataloader) + i

			# Configure model input
			imgs_hr = imgs["hr"].to(device)
			imgs_lr = imgs["lr"].to(device)

			# ------------------
			#  Train Generators
			# ------------------

			# optimize generator
			# discriminator.eval()
			for p in discriminator.parameters():
				p.requires_grad = False
			
			optimizer_G.zero_grad()
			
			gen_hr = generator(imgs_lr)

			# Scaling/Clipping output
			# gen_hr = gen_hr.clamp(0,1)
			
			if batches_done < args.warmup_batches:
				# Measure pixel-wise loss against ground truth
				if args.warmup_loss == "L1":
					loss_pixel = l1_loss(gen_hr, imgs_hr)
				elif args.warmup_loss == "L2":
					loss_pixel = l2_loss(gen_hr, imgs_hr)
				# Warm-up (pixel-wise loss only)
				loss_pixel.backward()
				optimizer_G.step()
				if args.debug:
					print("[Epoch %d/%d] [Batch %d/%d] [G pixel: %f]"
						% (epoch, args.epochs, i, len(dataloader), loss_pixel.item())
						)
				continue

			# Extract validity predictions from discriminator
			pred_real = discriminator(imgs_hr).detach()
			pred_fake = discriminator(gen_hr)

			# Adversarial ground truths
			valid = torch.ones_like(pred_real)
			fake = torch.zeros_like(pred_real)

			if args.gan == 'RAGAN':
				# Adversarial loss (relativistic average GAN)
				loss_GAN = bce_loss(pred_fake - pred_real.mean(0, keepdim=True), valid)
			elif args.gan == "VGAN":
				# Adversarial loss (vanilla GAN)
				loss_GAN = bce_loss(pred_fake, valid)

			# Content loss
			gen_features = feature_extractor(gen_hr)
			real_features = feature_extractor(imgs_hr).detach()
			if args.vgg_criterion == 'L1':
				loss_content = l1_loss(gen_features, real_features)
			elif args.vgg_criterion == 'L2':
				loss_content = l2_loss(gen_features, real_features)

			# For vgg hv loss ?? max-value
			# max_value = (1.1 * torch.max(torch.max(gen_features) , torch.max(real_features))).detach()
			# print(max_value , end = "\n\n")
			# loss_vgg_hv_psnr = hv_loss(1 - (psnr_fn(gen_features , real_features , max_value=max_value)/30) , 1 - ssim_fn_val(gen_features , real_features , max_value))

			psnr_val = psnr_fn(gen_hr , imgs_hr.detach())
			ssim_val = ssim_fn(gen_hr , imgs_hr.detach())


			# Total generator loss
			if args.method == 'M4':
				loss_hv_psnr = hv_loss(1 - (psnr_val/args.max_psnr) , 1 - ssim_val)
				loss_G = (loss_content * args.weight_vgg) + (loss_hv_psnr * args.weight_hv) + (args.weight_gan * loss_GAN)
			elif args.method == 'M1':
				loss_G = (loss_content * args.weight_vgg) + (args.weight_gan * loss_GAN)
			elif args.method == 'M5':
				psnr_loss = (1 - (psnr_val/args.max_psnr)).mean()
				ssim_loss = (1 - ssim_val).mean()
				loss_G = (loss_content * args.weight_vgg) + (args.weight_gan * loss_GAN) + (args.weight_pslinear * (ssim_loss + psnr_loss))
			elif args.method == 'M6':
				real_features_normalized = normalize_VGG_features(real_features)
				gen_features_normalized = normalize_VGG_features(gen_features)
				psnr_vgg_val = psnr_fn(gen_features_normalized , real_features_normalized)
				ssim_vgg_val = ssim_fn_vgg(gen_features_normalized , real_features_normalized)
				loss_vgg_hv = hv_loss(1 - (psnr_vgg_val/args.max_psnr) , 1 - ssim_vgg_val)
				loss_G = (args.weight_vgg_hv * loss_vgg_hv) + (args.weight_gan * loss_GAN)
			elif args.method == 'M7':
				loss_hv_psnr = hv_loss(1 - (psnr_val/args.max_psnr) , 1 - ssim_val)
				if (epoch - start_epoch) < args.loss_mem:
					loss_G = (loss_content * args.weight_vgg) + (loss_hv_psnr * args.weight_hv) + (args.weight_gan * loss_GAN)
				else:
					weight_vgg = (1/losses_log[-args.loss_mem: , 1].mean()) * args.mem_vgg_weight
					weight_bce = (1/losses_log[-args.loss_mem: , 2].mean()) * args.mem_bce_weight
					weight_hv = (1/losses_log[-args.loss_mem: , 3].mean()) * args.mem_hv_weight
					loss_G = (loss_content * weight_vgg) + (loss_hv_psnr * weight_hv) + (loss_GAN * weight_bce)				
			elif args.method == "M2":
	            loss_G = hv_loss(loss_GAN*args.weight_gan,loss_content*args.weight_vgg)			

			if args.include_l1:
				loss_G += (args.weight_l1 * l1_loss(gen_hr , imgs_hr))

			loss_G.backward()
			optimizer_G.step()
			scheduler_G.step()

			# ---------------------
			#  Train Discriminator
			# ---------------------

			# optimize discriminator
			# discriminator.train()
			for p in discriminator.parameters():
				p.requires_grad = True

			optimizer_D.zero_grad()

			pred_real = discriminator(imgs_hr)
			pred_fake = discriminator(gen_hr.detach())

			if args.gan == "RAGAN":
				# Adversarial loss for real and fake images (relativistic average GAN)
				loss_real = bce_loss(pred_real - pred_fake.mean(0, keepdim=True), valid)
				loss_fake = bce_loss(pred_fake - pred_real.mean(0, keepdim=True), fake)
			elif args.gan == "VGAN":
				# Adversarial loss for real and fake images (vanilla GAN)
				loss_real = bce_loss(pred_real, valid)
				loss_fake = bce_loss(pred_fake, fake)

			# Total loss
			loss_D = (loss_real + loss_fake) / 2

			if args.method == 'M7':
				if (epoch - start_epoch) >= args.loss_mem:
					weight_dis = (1/losses_log[-args.loss_mem: , 5].mean()) * args.mem_bce_weight
					loss_D = loss_D/weight_dis					

			loss_D.backward()
			optimizer_D.step()

			if args.method == "M4" or args.method == "M7":
				losses_gen += np.array([loss_content.item(), 
										loss_GAN.item(), 
										loss_hv_psnr.item(), 
										loss_G.item() , 
										loss_D.item(),
										psnr_val.mean().item(),
										ssim_val.mean().item(),
										])
			elif args.method == "M6":
				losses_gen += np.array([loss_content.item(), 
										loss_GAN.item(),
										psnr_vgg_val.mean().item(),
										ssim_vgg_val.mean().item(),
										loss_vgg_hv.item(), 
										loss_G.item() , 
										loss_D.item(),
										psnr_val.mean().item(),
										ssim_val.mean().item(),
										])
			else:
				losses_gen += np.array([loss_content.item(), 
										loss_GAN.item(), 
										loss_G.item() , 
										loss_D.item(),
										psnr_val.mean().item(),
										ssim_val.mean().item(),
										])

		losses_gen /= batch_count
		losses_gen = list(losses_gen)
		losses_gen.insert(0 , epoch)

		write_to_csv_file(os.path.join(args.output_path ,'train_log.csv' ), losses_gen)

		if (losses_log == np.zeros(loss_len+1)).sum() == loss_len+1:
			losses_log = np.expand_dims(np.array(losses_gen) , 0)
		else:
			losses_log = np.vstack((losses_log , losses_gen))

		if epoch%args.print_every == 0:
			print('Epoch' , epoch , 'Loss GAN :' , losses_gen)

		if epoch%args.plot_every == 0:
			plot_image(epoch , generator , test_dataloader)
			
		if epoch%args.test_every == 0:
			test(epoch , generator , test_dataloader) 

		if epoch%args.save_model_every == 0:
			checkpoint = {
				'epoch':epoch+1,
				'gen_state_dict':generator.state_dict(),
				'gen_optimizer_dict':optimizer_G.state_dict(),
				'gen_scheduler_dict':scheduler_G.state_dict(),
				'dis_state_dict':discriminator.state_dict(),
				'dis_optimizer_dict':optimizer_D.state_dict(),
			}
			os.makedirs(os.path.join(args.output_path , 'saved_model') ,exist_ok=True)
			torch.save(checkpoint, os.path.join(args.output_path , 'saved_model' ,'checkpoint_' + str(epoch) + ".pth"))

if __name__ == '__main__':
	train()

