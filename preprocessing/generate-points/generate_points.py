from numba import njit
from mpi4py import MPI
from shapely.geometry import Point, shape
import numpy as np
import random
import sys
import json

N = int(sys.argv[1])
random.seed(30123)

@njit
def generate_point(minx, miny, maxx, maxy):
    x = random.uniform(minx, maxx)
    y = random.uniform(miny, maxy)
    return (x, y)

def parallel_points_generation():
    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_workers = comm.Get_size()

    workload = N // n_workers

    if rank == 0:
        with open("../../data/subsahara_shapes.json", "r") as f:
            subsahara = json.load(f)
            subsahara['shape'] = shape(subsahara['shape'])
    else:
        subsahara = None

    subsahara = comm.bcast(subsahara, root=0)

    (minx, miny), (maxx, maxy) = subsahara['bbox']
    subsahara_shape = subsahara['shape']

    i = 0
    points = np.empty((workload, 2))
    while i < workload:
        x, y = generate_point(minx, miny, maxx, maxy)
        if subsahara_shape.contains(Point(x,y)):
            points[i,:] = (x,y)
            i += 1

    if rank == 0:
        all_points = np.empty((N,2))
    else:
        all_points = None

    comm.Gather(points, all_points, root=0)

    if rank == 0:
        np.savetxt(
            'points.csv',
            all_points,
            delimiter = ',',
            header = 'lon,lat'
        )

    return


if __name__ == "__main__":

    parallel_points_generation()
    
