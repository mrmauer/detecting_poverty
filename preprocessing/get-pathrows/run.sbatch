#!/bin/bash -l

#SBATCH --job-name=pathrows
#SBATCH --ntasks=16

conda activate geo
# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n 16 python3 ./match_pathrows.py 2> pathrow-error.log
