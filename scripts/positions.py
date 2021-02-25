# This file serves either as library for loading/saving/creating position data
# or (when used as main file) to create positions with args taken from the command line
import numpy as np
import xarray as xr
from pathlib import Path

# add heisensim path to sys.path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import heisensim as sim # pylint: disable=import-error
import scripts.simlib as simlib # pylint: disable=import-error


def empty_position_set(r_bl, realizations, system_size, dim=3):
    dataset = xr.DataArray(
        np.zeros((len(r_bl), realizations, system_size, dim), dtype=np.float64),
        dims=['r_bl', 'disorder_realization', 'particle', 'xyz'],
        coords={
            'r_bl': r_bl,
            'disorder_realization': np.arange(realizations),
            'particle': np.arange(system_size),
            'xyz': ['x', 'y', 'z']
        }
    )
    return dataset

## core function
def create_positions(geometry, N, disorder_realizations=100):
    r_bl_list = np.round(np.arange(0.2, 0.95, 0.05), 2)
    positions = empty_position_set(r_bl_list, disorder_realizations, N)

    for r_bl in r_bl_list:
        sampler = simlib.SAMPLING_GENERATORS[geometry](r_bl=r_bl, N=N) #sim.Sphere(r_bl=r_bl, radius=radius_from_packing(N=N), max_iter=int(1e5))
        for disorder_realization in range(disorder_realizations):
            pos = sampler.sample_positions(N)
            positions.loc[r_bl, disorder_realization] = pos
    return positions

## save/load functions
## abstracted to enable future optimization
def save_positions(data, path, *params):
    "Either provide a full path to a .nc file or a directory and geometry and N in params"
    if params:
        path = simlib.position_data_path(path, *params)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)

def load_positions(path, *params):
    if params:
        path = simlib.position_data_path(path, *params)
    return xr.load_dataarray(path)

## glue functions together
def main(path_prefix, seed, geometry, N, disorder_realizations):
    np.random.seed(seed)
    positions = create_positions(geometry, N, disorder_realizations)
    save_positions(positions, path_prefix, geometry, N)


if __name__ == '__main__':
    import argparse
    from pathlib import Path
    parser = argparse.ArgumentParser(description='Generate blockaded positions.')
    parser.add_argument("-p", "--path", type=Path, default=Path.cwd(), help="Data directory. Positiond data will be saved to 'positions' subdirectory")
    parser.add_argument("-r", "--realizations", metavar="n", type=int, help="Number of disorder realizations.", default=100)
    parser.add_argument("-s", "--seed", type=int, default=5, help="initial seed for the RNG")
    parser.add_argument("geometry", type=str, help="Geometry to sample from", choices=simlib.SAMPLING_GEOMETRIES)
    parser.add_argument("spins", type=int, help="Number of spins")
    args = parser.parse_args()

    main(args.path, args.seed, args.geometry, args.spins, args.realizations)


