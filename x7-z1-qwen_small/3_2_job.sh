#!/bin/bash

#SBATCH -J qwen-30B_left
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-08:00             # x min runtime
#SBATCH --gres=gpu:2
#SBATCH --mem=128G                  
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
# srun python 3_qwen.py "below" --sd_model "redcaps" --strategies inverse expert direct
# srun python 3_qwen.py "over" --sd_model "redcaps" --strategies inverse expert direct
# srun python 3_qwen.py "under" --sd_model "redcaps" --strategies inverse expert direct
# srun python 3_qwen.py "on top of" --sd_model "redcaps" --strategies inverse expert direct

srun python 3_qwen.py "behind" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "above" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "in front of" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "to the right of" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "to the left of" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "closeby" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "far from" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "without" --sd_model "cogvlm" --strategies inverse expert direct
srun python 3_qwen.py "between" --sd_model "cogvlm" --strategies inverse expert direct
