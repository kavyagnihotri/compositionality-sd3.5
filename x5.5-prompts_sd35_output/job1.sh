#!/bin/bash

# Slurm job script for Stable Diffusion comparison experiment
#SBATCH -J sd35_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-00:20             # 
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=256G                  # Increased memory for image generation
#SBATCH --output=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.out
#SBATCH --error=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

# Diagnostic Phase
echo "=== Job Information ==="
scontrol show job $SLURM_JOB_ID
echo -e "\n=== Working Directory ==="
pwd
echo -e "\n=== GPU Information ==="
nvidia-smi

ls $WORK

srun python sd35_.py --csv_file=/mnt/lustre/work/hein/hof217/stable_diffusion/x5.5-prompts_sd35_output/pizza.csv