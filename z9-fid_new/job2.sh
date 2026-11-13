#!/bin/bash
#SBATCH -J fid2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-00:30             # x hours runtime
#SBATCH --gres=gpu:1               # Using 1 GPUs
#SBATCH --mem=128G                  # Increased memory for image generation
#SBATCH --output=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.out
#SBATCH --error=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

source /usr/etc/profile.d/conda.sh
conda activate sd2

srun python3 4_fid.py --use-clean-fid