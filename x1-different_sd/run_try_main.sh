#!/bin/bash

# Sample Slurm job script for Galvani 

#SBATCH -J sdcomp                # Job name
#SBATCH --ntasks=1                 # Number of tasks
#SBATCH --cpus-per-task=8          # Number of CPU cores per task
#SBATCH --nodes=1                  # Ensure that all cores are on the same machine with nodes=1
#SBATCH --partition=2080-galvani   # Which partition will run your job
#SBATCH --time=0-05:00             # Allowed runtime in D-HH:MM
#SBATCH --gres=gpu:2               # (optional) Requesting type and number of GPUs
#SBATCH --mem=64G                  # Total memory pool for all cores (see also --mem-per-cpu); exceeding this number will cause your job to fail.
#SBATCH --output=/home/hein/hof217/project/stable_diffusion/sdcomp-%j.out       # File to which STDOUT will be written - make sure this is not on $HOME
#SBATCH --error=/home/hein/hof217/project/stable_diffusion/sdcomp-%j.err        # File to which STDERR will be written - make sure this is not on $HOME
#SBATCH --mail-type=ALL            # Type of email notification- BEGIN,END,FAIL,ALL
#SBATCH --mail-user=kavyagnihotri17@gmail.com   # Email to which notifications will be sent

# Diagnostic and Analysis Phase - please leave these in.
scontrol show job $SLURM_JOB_ID
pwd
nvidia-smi # only if you requested gpus
ls $WORK # not necessary just here to illustrate that $WORK is available here

# Setup Phase
# add possibly other setup code here, e.g.
# - copy singularity images or datasets to local on-compute-node storage like /scratch_local
# - loads virtual envs, like with anaconda
# - set environment variables
# - determine commandline arguments for `srun` calls
conda activate sd2

# Compute Phase
srun python3 try_main.py \
    --output_dir "$WORK/project/stable_diffusion" \
    --batch_size 4  
