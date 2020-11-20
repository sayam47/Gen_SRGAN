import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

def get_pd(file_name = '/home/sayam/Desktop/Prev Sem/BTP2.0/out_loss_256_M4(600-900).csv'):
	b = pd.read_csv(file_name , header = None , names = ['epoch' , 'vgg_loss' , 'bce_loss' , 'gan_total' , 'dis_loss', 'psnr' , 'ssim'])
	b['mse'] = 1/(10**(b['psnr']/10))
	b['ssim_loss'] = 1 - b['ssim']
	b['psnr_loss'] = 1 - (b['psnr']/35.5)
	b['hv_loss'] = 1.1 * np.max((b['psnr_loss'].values , b['ssim_loss'].values) , axis = 0)
	b['hv_loss'] = -np.log(b['hv_loss'] - b['psnr_loss']) - np.log(b['hv_loss'] - b['ssim_loss'])
	return b


def plot_pd(b , col_names = ['vgg_loss' , 'bce_loss' , 'hv_loss'] , mul = [1 , 0.001 , 0.0005] , offset = 0):
	xb = np.arange(0,len(b))
	for i,name in enumerate(col_names):
		plt.plot(xb[offset:] , b[name][offset:]*mul[i] , label = name)
	plt.legend()
	plt.grid()
	plt.show()


def plot_normal(arrays , col_names , offset = 0 , save = 0 , filename = 'psnr_plot.png'):
	for i,name in enumerate(col_names):
		xb = np.arange(0,len(arrays[i]))
		plt.plot(xb[offset:] , arrays[i][offset:] , label = col_names[i])
	plt.legend()
	plt.grid()
	if save:
		plt.savefig(filename)
		plt.close()
		return 
	plt.show()
	plt.close()

M1 = get_pd("/home/sayam/results_output_xin_esrgan/train_log.csv")
M2 = get_pd("/home/sayam/results_output_M2_2/train_log.csv")
M4 = get_pd("/home/sayam/results_output_M4_999_esrgan/train_log.csv")
M5 = get_pd("/home/sayam/results_output_M5_999_esrgan/train_log.csv")
M6 = get_pd("/home/sayam/results_output_M7_esrgan_equal/train_log.csv")

plot_normal([M1['psnr'] , M2['psnr'] , M4['psnr'] , M5['psnr'] , M6['psnr']] , ['M1' , 'M2' , 'M4' , 'M5' , 'M6'] , offset=50 ,  save=1 , filename='psnr_plot.png')
plot_normal([M1['ssim'] , M2['ssim'] , M4['ssim'] , M5['ssim'] , M6['ssim']] , ['M1' , 'M2' , 'M4' , 'M5' , 'M6'] , offset=50 ,  save=1 , filename='ssim_plot.png')