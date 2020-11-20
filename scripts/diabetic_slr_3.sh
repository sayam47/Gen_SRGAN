#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=diabetic_retinopathy_slr_3
#SBATCH --error=diabetic_slr_3.err
#SBATCH --output=diabetic_slr_3.out
#SBATCH --partition=gpu
#SBATCH --gres=gpu:2


module load cuda/10.1
source /home/sayam.choudhary.cse17.iitbhu/HNAS-SR-master/src/env/bin/activate
epochs=210
plot_every=5
print_every=1
debug=0
batch_size=16
n_threads=16
train_data_path="/scratch/sayam.choudhary.cse17.iitbhu/diabetic_retinopathy_1/resized_train/resized_train"
test_data_path="/scratch/sayam.choudhary.cse17.iitbhu/diabetic_retinopathy_1/resized_train/test_set"
output_path="/scratch/sayam.choudhary.cse17.iitbhu/output_diabetic_slr3"

# VGG(L1) * 1 + RAGAN * 0.0225 + HV(PSNR/35.5) * 0.09
python3 train.py --train_data_path "$train_data_path" --test_data_path "$test_data_path" --epochs $epochs --plot_every $plot_every --print_every $print_every --n_threads $n_threads --output_path "$output_path" --batch_size $batch_size --debug $debug --output_size 256x256 --warmup_batches 0 --method M4 --gan RAGAN --vgg_criterion L1 --weight_gan 0.0225 --weight_vgg 1 --weight_hv 0.09 --max_psnr 45.5 --betas 0.9,0.999 --lr 0.000007
