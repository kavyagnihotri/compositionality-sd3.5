#!/bin/bash

# Slurm job script for full custom SD3.5 workflow (conversion + testing)
#SBATCH -J full_custom_sd35
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-01:00             # 60+00 minutes runtime
#SBATCH --gres=gpu:1               # Request 1 GPU
#SBATCH --mem=128G                 # Memory for image generation
#SBATCH --output=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.out
#SBATCH --error=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

# ---------------------------
# Diagnostic Phase
# ---------------------------
echo "=== Job Information ==="
scontrol show job $SLURM_JOB_ID
echo -e "\n=== Working Directory ==="
pwd
echo -e "\n=== GPU Information ==="
nvidia-smi

# ---------------------------
# Setup Phase
# ---------------------------
conda activate sd2

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install diffusers transformers accelerate open-clip-torch numpy Pillow

# Navigate to project directory
cd /mnt/lustre/work/hein/hof217/stable_diffusion

# Create output directories
mkdir -p logs test_outputs converted_text_encoder

# ---------------------------
# Compute Phase - Full Workflow
# ---------------------------
echo -e "\n=== Running Full Custom SD3.5 Workflow ==="
srun python3 run_full_workflow.py --csv_file sept_w4/misc_words_prompts.csv


echo -e "\n=== Workflow Complete ==="
echo "Converted model: /mnt/lustre/work/hein/hof217/stable_diffusion/converted_text_encoder"
