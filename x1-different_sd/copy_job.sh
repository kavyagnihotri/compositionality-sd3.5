#!/bin/bash

# Slurm job script for Stable Diffusion comparison experiment
#SBATCH -J sd_comparison
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-08:00             # 8 hours runtime (adjust based on expected duration)
#SBATCH --gres=gpu:1               
#SBATCH --mem=128G                  # Increased memory for image generation
#SBATCH --output=/home/hein/hof217/project/stable_diffusion/logs/job-%j.out
#SBATCH --error=/home/hein/hof217/project/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de
#SBATCH --partition=hein

# ---------------------------
# Diagnostic Phase
# ---------------------------
echo "=== Job Information ==="
scontrol show job $SLURM_JOB_ID
echo -e "\n=== Working Directory ==="
pwd
echo -e "\n=== GPU Information ==="
nvidia-smi

ls $WORK

# ---------------------------
# Setup Phase
# ---------------------------
conda activate sd2

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate datasets pandas scikit-learn tqdm

# Create output directories
mkdir -p /home/hein/hof217/project/stable_diffusion/{logs,outputs}

# ---------------------------
# Compute Phase
# ---------------------------
echo -e "\n=== Starting Experiment ==="
srun python3 copy_sd_comp.py \
    --output_dir /scratch/$USER/sd_experiment/outputs \
    --samples_per_category 5 \
    --num_images_per_prompt 1

