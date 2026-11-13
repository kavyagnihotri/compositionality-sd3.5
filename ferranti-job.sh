#!/bin/bash

# Slurm job script for Qwen3-VL-235B-A22B-Instruct experiment
#SBATCH -J qwen-30B
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=h100-ferranti     
#SBATCH --time=0-01:00               # 2 hour runtime (adjust as needed)
#SBATCH --gres=gpu:4                 
#SBATCH --mem=256G                   
#SBATCH --output=/home/hein/hof217/qwen/logs/job-%j.out  # *** CHANGED: Adjusted log path for Qwen3 ***
#SBATCH --error=/home/hein/hof217/qwen/logs/job-%j.err   # *** CHANGED: Adjusted log path for Qwen3 ***
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
conda activate qwen # Assuming the same environment has necessary dependencies

# Create output directories
mkdir -p /home/hein/hof217/project/qwen/logs 

# ---------------------------
# Compute Phase
# ---------------------------
echo -e "\n=== Starting Experiment for Qwen3-VL-235B-A22B-Instruct ==="
# Ensure qwen.py is updated to use the new model processor/tokenizer
srun python qwen.py