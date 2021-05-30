from mpi4py import MPI
from shapely.geometry import Point
import sys
import geopandas as gpd

# Use all points data or only a user-input subset
if len(sys.argv) > 1:
    N = int(sys.argv[1])
else:
    N = 1e6

def parallel_pathrow_matching():
    # Get rank of process and overall size of communicator:
    comm = MPI.COMM_WORLD
    rank = comm.Get_rank()
    n_workers = comm.Get_size()
    workload = N // n_workers

    if rank == 0:
        # subset shapes to subsaharan africa 
        subsahara = gpd.read_file('../../data/subsahara.geojson').geometry[0]
        wrs = gpd.read_file('landsat-path-row/WRS2_descending.shp')
        wrs_subsahara = wrs.loc[
            wrs.geometry.map(subsahara.intersects),
            ['PATH', 'ROW', 'geometry']
            ].copy()

        # create Rtree
        wrs_rtree = wrs_subsahara.sindex

        # read in points
        all_points = np.loadtxt(
            '../generate-points/points.csv',
            skiprows = 1
        )
    else:
        wrs_subsahara = None
        wrs_rtree = None
        all_points = None

    wrs_subsahara = comm.bcast(wrs_subsahara, root=0)
    wrs_rtree = comm.bcast(wrs_rtree, root=0)

    points = np.empty((workload, 2))
    comm.Scatter(all_points, points, root=0)

    pathrows = np.empty_like(points)

    i = 0
    while i < workload:
        x, y = points[i,:]
        match = wrs_rtree.query(Point(x,y), predicate='within').item()
        if match:
            pathrows[match,:] = wrs_subsahara.PATH[match], wrs_subsahara.ROW[match]
        else:
            pathrows[match,:] = (-1, -1)
        i += 1

    pathrows = np.concatenate([points, pathrows], axis=1)

    if rank == 0:
        all_pathrows = np.empty((N,4))
    else:
        all_pathrows = None

    comm.Gather(pathrows, all_pathrows, root=0)

    if rank == 0:
        np.savetxt(
            'point2pathrow.csv',
            all_pathrows,
            delimiter = ',',
            header = 'lon,lat,path,row'
        )

    return


if __name__ == "__main__":
    parallel_pathrow_matching()
    
