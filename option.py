import argparse

parser = argparse.ArgumentParser(description='Super Resolution')

parser.add_argument('--model', type=str , default='rrdb' , help = 'architecture of generator')
parser.add_argument('--loss' , type=str , default='1*GAN' , help = 'loss function for training')
parser.add_argument('--epochs' , type=int ,default = 8000 , help= 'number of epochs')
parser.add_argument('--print_every' , type=int , default = 10 , help = 'print loss value after every n epochs')
parser.add_argument('--plot_every' , type=int , default = 10 , help = 'plot output images after every n epochs')
parser.add_argument('--train_data_path' , type=str , default='/home/sayam.choudhary.cse17.iitbhu/sr_df2k/' , help='location of training images')
parser.add_argument('--test_data_path' , type=str , default = '/home/sayam.choudhary.cse17.iitbhu/sr_test/' , help='location of testing images')
parser.add_argument('--output_path' ,type=str , default = '/scratch/sayam.choudhary.cse17.iitbhu/output/' , help='location of output')
parser.add_argument('--output_size' ,type=str , default = '128x128' , help='size of output image')
parser.add_argument('--n_threads', type=int, default=6,help='number of threads for data loading')
parser.add_argument('--debug' , type=int , default=0 ,help='turn debug on')
parser.add_argument('--save_model_every' ,type=int , default = 300 , help = 'save model params every n iterations')
parser.add_argument('--resume_training' ,type=int , default = 0 , help = 'resume training using checkpoint')
parser.add_argument('--checkpoint_path' ,type=str , default = '/scratch/sayam.choudhary.cse17/output/saved_model/checkpoint_0.pth' , help = 'save model params every n iterations')
parser.add_argument('--scale' , type=int ,default = 4 , help='x scale upscaling')
parser.add_argument('--patch_size' , type=str , default='128x128' , help='patch size of training image')
parser.add_argument('--batch_size' , type=int ,  default = 16 , help = 'batch size for training')
parser.add_argument('--test_every' ,  type=int , default = 1 , help = 'test every n epochs')
parser.add_argument('--plot_test' , type=int , default = 1 , help = 'plot tested image')
parser.add_argument('--warmup_batches' , type=int , default = 0 , help = 'warmup batches of L1/L2 loss before original loss')
parser.add_argument('--warmup_loss' , type=str , default = 'L1' , help = 'warmup loss criteriion')
parser.add_argument('--lr' , type=float , default = 0.0001 , help = 'learning rate')
parser.add_argument('--betas' , type=str , default = "0.9,0.99" , help = 'betas for adam')
parser.add_argument('--multistep_lr' , type=int , default = 1, help = 'use multistep lr scheduler')
parser.add_argument('--multistep_milestones' , type=str , default = "5000,100000,200000,300000", help = 'milestones for multistep lr scheduler')
parser.add_argument('--multistep_gamma' , type=float , default = 0.5, help = 'gamma for multistep lr scheduler')
parser.add_argument('--discriminator', type=str , default='vgg128' , help = 'architecture of discriminator')
parser.add_argument('--method', type=str , default='M4' , help = 'loss function model')
parser.add_argument('--max_psnr', type=float , default=30 , help = 'max psnr for hv loss')
parser.add_argument('--gan', type=str , default="RAGAN" , help = 'adversarial loss type')
parser.add_argument('--weight_gan', type=float , default=0.001 , help = 'adversarial loss weight')
parser.add_argument('--weight_vgg', type=float , default=1 , help = 'vgg loss weight')
parser.add_argument('--vgg_criterion', type=str , default='L1' , help = 'vgg loss criteriion')
parser.add_argument('--weight_hv', type=float , default=0.0005 , help = 'hv_loss weight')
parser.add_argument('--include_l1', type=int , default=0 , help = 'include l1 pixel loss')
parser.add_argument('--weight_l1', type=float , default=0.01 , help = 'l1 loss weight')


# after upload changes
parser.add_argument('--weight_pslinear', type=float , default=0.05 , help = 'psnr ssim loss weight')
parser.add_argument('--weight_vgg_hv', type=float , default=1 , help = 'weight of vgg hv loss')
parser.add_argument('--loss_mem', type=int , default=10 , help = 'length of loss memory used to take mean')
parser.add_argument('--mem_vgg_weight', type=float , default=1 , help = 'weight of loss memory vgg')
parser.add_argument('--mem_hv_weight', type=float , default=0.25 , help = 'weight of loss memory hv')
parser.add_argument('--mem_bce_weight', type=float , default=0.75 , help = 'weight of loss memory bce')



args = parser.parse_args()

args.debug = bool(args.debug)
args.resume_training = bool(args.resume_training)
args.plot_test = bool(args.plot_test)
args.multistep_lr = bool(args.multistep_lr)
args.include_l1 = bool(args.include_l1)

args.patch_size = tuple(map(int ,args.patch_size.split('x')))
args.output_size = tuple(map(int ,args.output_size.split('x')))
args.loss = list(map(lambda x:(int(x[0]),x[1]) , list(map(lambda x:x.split('*') , args.loss.split('+')))))
args.multistep_milestones = list(map(int , args.multistep_milestones.split(',')))
args.betas = tuple(map(float , args.betas.split(',')))

if __name__ == '__main__':
	print(args)
