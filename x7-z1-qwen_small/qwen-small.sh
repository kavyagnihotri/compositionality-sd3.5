#!/bin/bash

#SBATCH -J qwen-30B-all-leftover
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-02:00             # 2 hour runtime
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
conda activate qwen

# Create output directories
mkdir -p /home/hein/hof217/project/stable_diffusion/logs

# ---------------------------
# Compute Phase
# ---------------------------
echo -e "\n=== Starting Experiment ==="
srun python qwen-small-all.py "behind"
srun python qwen-small-all.py "in front of"
srun python qwen-small-all.py "to the left of"
srun python qwen-small-all.py "to the right of"
