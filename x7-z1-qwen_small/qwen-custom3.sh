#!/bin/bash

#SBATCH -J qwen-between-rest
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-00:40             # 6 hour runtime
#SBATCH --gres=gpu:2
#SBATCH --mem=256G                  
#SBATCH --output=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.out
#SBATCH --error=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

source /usr/etc/profile.d/conda.sh
conda activate qwen

echo -e "\n=== Starting Experiment ==="

