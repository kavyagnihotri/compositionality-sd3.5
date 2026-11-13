#!/bin/bash

# Slurm job script for Stable Diffusion comparison experiment
#SBATCH -J llava_1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=4
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-02:00             # 2 hour runtime (adjust based on expected duration)
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=64G                  # Increased memory for image generation
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
conda activate llava

# # Install dependencies
# pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
# pip install diffusers transformers accelerate datasets pandas scikit-learn tqdm

# Create output directories
mkdir -p /home/hein/hof217/project/stable_diffusion/logs

# ---------------------------
# Compute Phase
# ---------------------------
echo -e "\n=== Starting Experiment ==="
srun python llava_vlm_context_learning.py "below" --examples examples.csv