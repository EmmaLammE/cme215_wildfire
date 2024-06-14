#!/bin/bash
#SBATCH --job-name=multifire
#SBATCH --time=20:00:00
#SBATCH --mem-per-cpu=250GB
#SBATCH -p serc
#SBATCH --cpus-per-task=1
#SBATCH --ntasks=4
#SBATCH -o ./../sbatch_output_logs/out_train_multifire.%j.out
#SBATCH -e ./../sbatch_output_logs/err_train_multifire.%j.err

# below you run/call your code, load modules, python, Matlab, R, etc.
# and do any other scripting you want
# lines that begin with #SBATCH are directives (requests) to the scheduler-SLURM module load python/3.6.1
mkdir ../sbatch_output_logs
python3 train_fire.py