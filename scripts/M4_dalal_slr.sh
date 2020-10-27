#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=M4_dalal_slr
#SBATCH --error=M4_dalal_slr.err
#SBATCH --output=M4_dalal_slr.out
#SBATCH --partition=gpu


module load cuda/10.1
source /home/sayam.choudhary.cse17.iitbhu/HNAS-SR-master/src/env/bin/activate
epochs=2101
plot_every=1
print_every=1
debug=0
batch_size=16
n_threads=16
train_data_path="/home/sayam.choudhary.cse17.iitbhu/sr_df2k"
test_data_path="/home/sayam.choudhary.cse17.iitbhu/Set5"
output_path="/scratch/sayam.choudhary.cse17.iitbhu/output_dalal_slr"

# VGG(L1) * 1 + RAGAN * 0.0225 + HV(PSNR/35.5) * 0.09
python3 train.py --train_data_path "$train_data_path" --test_data_path "$test_data_path" --epochs $epochs --plot_every $plot_every --print_every $print_every --n_threads $n_threads --output_path "$output_path" --batch_size $batch_size --debug $debug --output_size 256x256 --warmup_batches 0 --method M4 --gan RAGAN --vgg_criterion L1 --weight_gan 0.0225 --weight_vgg 1 --weight_hv 0.09 --max_psnr 35.5 --betas 0.9,0.99 --lr 0.00005
