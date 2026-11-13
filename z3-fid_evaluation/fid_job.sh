#!/bin/bash
#SBATCH -J fid_generate_image_then_eval
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-05:00             # 10 hours runtime
#SBATCH --gres=gpu:1               # Using 1 GPUs
#SBATCH --mem=128G                  # Increased memory for image generation
#SBATCH --output=/home/hein/hof217/project/stable_diffusion/logs/job-%j.out
#SBATCH --error=/home/hein/hof217/project/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

source /usr/etc/profile.d/conda.sh
conda activate sd2

# srun python3 0_download_coco_captions.py
# srun python3 1_prepare_coco_ground_truth.py
srun python3 5_cogvlm_generate_and_fid.py

