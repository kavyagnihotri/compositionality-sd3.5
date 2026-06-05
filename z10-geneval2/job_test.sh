#!/bin/bash
#SBATCH -J geneval2_test
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-02:00
#SBATCH --gres=gpu:1
#SBATCH --mem=64G
#SBATCH --output=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.out
#SBATCH --error=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

cd /mnt/lustre/work/hein/hof217/stable_diffusion
export HF_HOME=/mnt/lustre/work/hein/hof217/.cache/huggingface

srun /home/hein/hof217/.conda/envs/sd2/bin/python z10-geneval2/run_geneval2.py \
    --model_variant standard \
    --max_prompts 20 \
    --output_dir z10-geneval2/outputs/standard_28steps_test
