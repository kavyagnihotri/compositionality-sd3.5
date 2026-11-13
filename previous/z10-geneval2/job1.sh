#!/bin/bash
#SBATCH -J geneval2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=1-12:00             # enough for full GenEval2 generation + Soft-TIFA evaluation
#SBATCH --gres=gpu:1               # Using 1 GPU
#SBATCH --mem=128G                 # Image generation and VLM evaluation are memory-heavy
#SBATCH --output=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.out
#SBATCH --error=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

cd /mnt/lustre/work/hein/hof217/stable_diffusion

source /mnt/lustre/work/hein/hof217/stable_diffusion/.venv/bin/activate

# Full GenEval2 run with the local stable-diffusion-3.5-large-turbo checkpoint.
# This generates all prompts, saves the prompt->image mapping, and runs Soft-TIFA.
srun python3 z10-geneval2/run_sd35_geneval2.py