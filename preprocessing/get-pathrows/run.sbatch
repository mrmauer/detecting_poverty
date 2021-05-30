#!/bin/bash

#SBATCH --job-name=pathrows
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --ntasks=4

# Load Python and MPI modules
module load python/anaconda-2019.03
module load intelmpi/2018.2.199+intel-18.0


# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n $i python ./get_pathrows.py 10000