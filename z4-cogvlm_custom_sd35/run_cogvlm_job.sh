#!/bin/bash

# Slurm job script for CogVLM custom SD3.5 workflow (conversion + testing)
#SBATCH -J cogvlm_custom_sd35
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --nodes=1
#SBATCH --partition=a100-galvani
#SBATCH --time=0-02:30
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

# Uncomment the line below and comment out the line above to use CSV
srun python3 run_full_workflow.py --csv_file ../x5.5-prompts_sd35_output/simple_positional_01.csv --skip_default_tests
srun python3 run_full_workflow.py --csv_file ../x5.5-prompts_sd35_output/simple_positional_4181917.csv --skip_default_tests
srun python3 run_full_workflow.py --csv_file ../x5.5-prompts_sd35_output/simple_positional_49993168.csv --skip_default_tests
srun python3 run_full_workflow.py --csv_file ../x5.5-prompts_sd35_output/simple_positional_71360340.csv --skip_default_tests
srun python3 run_full_workflow.py --csv_file ../x5.5-prompts_sd35_output/misc_words_prompts.csv --skip_default_tests


