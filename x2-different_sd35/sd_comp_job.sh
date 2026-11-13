#!/bin/bash

# Slurm job script for Stable Diffusion comparison experiment
#SBATCH -J sd_comp
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-04:00             # 4 hours runtime (adjust based on expected duration)
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=256G                  # Increased memory for image generation
#SBATCH --output=/home/hein/hof217/project/stable_diffusion/logs/job-%j.out
#SBATCH --error=/home/hein/hof217/project/stable_diffusion/logs/job-%j.err
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
conda activate sd2

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets pandas scikit-learn tqdm

# Create output directories
mkdir -p /home/hein/hof217/project/stable_diffusion/logs

# ---------------------------
# Compute Phase
# ---------------------------
echo -e "\n=== Starting Experiment ==="
srun python3 sd_comp.py \
    --samples_per_category 20 \
    --num_images_per_prompt 1