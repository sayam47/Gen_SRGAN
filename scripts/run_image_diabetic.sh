#!/bin/sh
#SBATCH -N 1
#SBATCH --ntasks-per-node=40
#SBATCH --job-name=img_result
#SBATCH --error=img_result.err
#SBATCH --output=img_result.out
#SBATCH --partition=cpu


module load cuda/10.1
source /home/sayam.choudhary.cse17.iitbhu/HNAS-SR-master/src/env/bin/activate

python3 get_image_diabetic.py --debug 1
