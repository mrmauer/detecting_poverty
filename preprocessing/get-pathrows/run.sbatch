#!/bin/bash

#SBATCH --job-name=pathrows
#SBATCH --partition=broadwl
#SBATCH --constraint=fdr
#SBATCH --ntasks=2

# Load Python and MPI modules
# module load python/anaconda-2019.03
# module load intelmpi/2018.2.199+intel-18.0
module load python/anaconda-2020.02
module load intelmpi/5.1+intel-16.0

# Install Geopandas and dependencies
pip install fiona shapely rtree pyproj descartes geopandas

# Run the python program with mpirun, using & to run jobs at the same time
mpirun -n 2 python ./match_pathrows.py 100 2> pathrow-error.log
