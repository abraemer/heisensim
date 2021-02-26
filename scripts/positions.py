# This file serves either as library for loading/saving/creating position data
# or (when used as main file) to create positions with args taken from the command line
from pathlib import Path

import numpy as np
import xarray as xr

import simlib


def empty_position_set(rho, realizations, system_size, dim=3):
    dataset = xr.DataArray(
        np.zeros((len(rho), realizations, system_size, dim), dtype=np.float64),
        dims=['rho', 'disorder_realization', 'particle', 'xyz'],
        coords={
            'rho': rho,
            'disorder_realization': np.arange(realizations),
            'particle': np.arange(system_size),
            'xyz': np.arange(dim)
        }
    )
    return dataset

## core function
def create_positions(geometry, dim, N, disorder_realizations=100):
    rho_list = np.round(np.arange(0.2, 0.95, 0.05), 2)
    positions = empty_position_set(rho_list, disorder_realizations, N, dim)

    for rho in rho_list:
        sampler = simlib.SAMPLING_GENERATORS[geometry](rho=rho, N=N, dim=dim)
        for disorder_realization in range(disorder_realizations):
            pos = sampler.sample_positions(N)
            positions.loc[rho, disorder_realization] = pos
    return positions

## save/load functions
## abstracted to enable future optimization
def save_positions(data, path, *params, **kwargs):
    "Either provide a full path to a .nc file or a directory and geometry and N in params"
    if params:
        path = simlib.position_data_path(path, *params, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)

def load_positions(path, *params, **kwargs):
    if params:
        path = simlib.position_data_path(path, *params, **kwargs)
    return xr.load_dataarray(path)

## glue functions together
def main(path_prefix, force, seed, geometry, dim, N, disorder_realizations):
    np.random.seed(seed)
    save_path = simlib.position_data_path(path_prefix, geometry, dim, N)
    if not force and save_path.exists():
        # check amount of realizations?
        print(f"Positions with params: geometry={geometry}, dim={dim}, N={N} already exist. Skipping. Use --force to overwrite.")
        exit()
    positions = create_positions(geometry, dim, N, disorder_realizations)
    save_positions(positions, save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate blockaded positions.')
    parser.add_argument("-p", "--path", type=Path, default=Path.cwd(), help="Data directory. Positiond data will be saved to 'positions' subdirectory")
    parser.add_argument("-r", "--realizations", metavar="n", type=int, help="Number of disorder realizations.", default=100)
    parser.add_argument("-d", "--dimensions", metavar="d", type=int, help="Number of spatial dimensions (1,2,3 are supported)", default=3)
    parser.add_argument("-s", "--seed", type=int, default=5, help="initial seed for the RNG")
    parser.add_argument("-F", "--force", action="store_true", help="Force overwriting of existing data.")
    parser.add_argument("geometry", type=str, help="Geometry to sample from", choices=simlib.SAMPLING_GEOMETRIES)
    parser.add_argument("spins", type=int, help="Number of spins")
    args = parser.parse_args()

    main(args.path, args.force, args.seed, args.geometry, args.dimensions, args.spins, args.realizations)


