#!/bin/bash

#SBATCH -J qwen-30B-not_without
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-00:45             # 45 min runtime
#SBATCH --gres=gpu:2
#SBATCH --mem=256G                  
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

# Setup Phase
source /usr/etc/profile.d/conda.sh
conda activate qwen

echo -e "\n=== Starting Experiment ==="
srun python qwen-small-reason-custom-sd35.py "and not"
srun python qwen-small-reason-custom-sd35.py "without"
