#!/bin/bash

# Slurm job script for Stable Diffusion comparison experiment
#SBATCH -J sd35_modified_prompt
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-02:00             
#SBATCH --gres=gpu:1               
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
srun python ./sept_w4/sd35_prompt.py --csv_file=/mnt/lustre/work/hein/hof217/stable_diffusion/sept_w4/simple_positional_4181917.csv  --phrase="above"