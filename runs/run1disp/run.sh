#!/bin/bash

#SBATCH -N 1
#SBATCH --ntasks=1
#SBATCH --mem=64GB
#SBATCH --cpus-per-task=16
#SBATCH --error=/pbs/home/a/astropart03/hackaton/Class_models/run1disp/run.err
#SBATCH --output=/pbs/home/a/astropart03/hackaton/Class_models/run1disp/run.out
#SBATCH --time=3:00:00
#SBATCH --partition=htc

python3 -u run.py > run_out
