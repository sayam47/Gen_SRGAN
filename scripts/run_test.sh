#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=test_result
#SBATCH --error=test_result.err
#SBATCH --output=test_result.out
#SBATCH --partition=cpu


module load cuda/10.1
source /home/sayam.choudhary.cse17.iitbhu/HNAS-SR-master/src/env/bin/activate

test_base_path="/scratch/sayam.choudhary.cse17.iitbhu/SR_testing_datasets/"
list_datasets="$(ls $test_base_path)"

debug=1
batch_size=1

output_path="/home/sayam.choudhary.cse17.iitbhu/test_output"

#Model name and Path
#model_path="/scratch/sayam.choudhary.cse17.iitbhu/output_M4_l1_esrgan/saved_model/checkpoint_2100.pth"
#model_name="M4_l1"

#model_path="/scratch/sayam.choudhary.cse17.iitbhu/output_M7_esrgan_equal/saved_model/checkpoint_1800.pth"
#model_name="M7_esrgan_equal"

#model_path="/scratch/sayam.choudhary.cse17.iitbhu/output_xin_esrgan/saved_model/checkpoint_2100.pth"
#model_name="M1_xin"

#model_path="/scratch/sayam.choudhary.cse17.iitbhu/output_M5_999_esrgan/saved_model/checkpoint_2100.pth"
#model_name="M5"

model_path="/scratch/sayam.choudhary.cse17.iitbhu/output_M4_999_esrgan/saved_model/checkpoint_2100.pth"
model_name="output_M4_999_esrgan"

for dataset_name in $list_datasets
do 
test_data_path="$test_base_path$dataset_name/"
python3 test_result.py --test_data_path "$test_data_path" --output_path "$output_path" --batch_size $batch_size --debug $debug --dataset_name "$dataset_name" --model_path "$model_path" --model_name "$model_name" 
done
