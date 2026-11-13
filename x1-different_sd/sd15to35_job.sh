#!/bin/bash

# Slurm job script for Stable Diffusion comparison experiment
#SBATCH -J sd_comparison
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=2080-galvani
#SBATCH --time=0-08:00             # 8 hours runtime (adjust based on expected duration)
#SBATCH --gres=gpu:1               # Using 2 GPUs (for parallel SD1.5/SD3.5 execution)
#SBATCH --mem=256G                  # Increased memory for image generation
#SBATCH --output=/home/hein/hof217/project/stable_diffusion/logs/job-%j.out
#SBATCH --error=/home/hein/hof217/project/stable_diffusion/logs/job-%j.err
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

ORIGINAL_OUTPUT_DIR="/home/hein/hof217/project/stable_diffusion/output_20250805_210717"  # Replace with actual path

# Create new output directory for SD3.5
NEW_OUTPUT_DIR="${ORIGINAL_OUTPUT_DIR}_sd35"
mkdir -p $NEW_OUTPUT_DIR

echo -e "\n=== Starting Experiment ==="
srun python3 sd15_to_35_medium.py \
    --tensor_dir $ORIGINAL_OUTPUT_DIR \
    --output_dir $NEW_OUTPUT_DIR \
    --num_images_per_prompt 1 \
    --hf_token $HUGGINGFACE_TOKEN


