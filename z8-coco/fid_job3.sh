#!/bin/bash
#SBATCH -J fid_generate_image_then_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-24:00             # 24 hours runtime
#SBATCH --gres=gpu:1               # Using 1 GPUs
#SBATCH --mem=128G                  # Increased memory for image generation
#SBATCH --output=/home/hein/hof217/project/stable_diffusion/logs/job-%j.out
#SBATCH --error=/home/hein/hof217/project/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

source /usr/etc/profile.d/conda.sh
conda activate sd2

# 5k * 3 = 15k images to be generated and evaluated
# 100 images take 25 mins so 15k images will take 25*150 = 3750 mins = 62.5 hours. ?

srun python3 3_generate_images.py --model cogvlm

