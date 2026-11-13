#!/bin/bash

# Slurm job script for CogVLM custom SD3.5 workflow (conversion + testing)
#SBATCH -J cogvlm_custom_sd35_2
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-02:00
#SBATCH --gres=gpu:1
#SBATCH --mem=128G
#SBATCH --output=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.out
#SBATCH --error=/mnt/lustre/work/hein/hof217/stable_diffusion/logs/job-%j.err
#SBATCH --mail-type=ALL
#SBATCH --mail-user=kavyanjali.agnihotri@student.uni-tuebingen.de

echo "Job Information"
scontrol show job $SLURM_JOB_ID
echo ""
echo "Working Directory"
pwd
echo ""
echo "GPU Information"
nvidia-smi

source /usr/etc/profile.d/conda.sh
conda activate sd2

cd /mnt/lustre/work/hein/hof217/stable_diffusion/z4-cogvlm_custom_sd35
mkdir -p logs

echo ""
echo "Running CogVLM Custom SD3.5 Workflow"

srun python3 run_full_workflow.py --csv_file ../x5.5-prompts_sd35_output/negate_words_prompts.csv --skip_default_tests
srun python3 run_full_workflow.py --csv_file ../x5.5-prompts_sd35_output/negative_words_prompts.csv --skip_default_tests
srun python3 run_full_workflow.py --csv_file ../z1-new-prompts/misc_99165446.csv --skip_default_tests
srun python3 run_full_workflow.py --csv_file ../z1-new-prompts/single_class_92201291.csv --skip_default_tests