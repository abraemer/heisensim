# This file serves either as library for loading/saving/creating position data
# or (when used as main file) to create positions with args taken from the command line
from pathlib import Path

import numpy as np
import xarray as xr

import sys
sys.path.append(str(Path(__file__).parent.parent))
import heisensim as sim # pylint: disable=import-error
import scripts.simlib as simlib # pylint: disable=import-error
import scripts.run_ed as edlib # pylint: disable=import-error


def empty_result_set(data):
    dataset = xr.DataArray(
        np.zeros((3, len(data.rho), len(data.h), len(data.disorder_realization)), dtype=np.float64),
        dims=['ensemble', 'rho', 'h', 'disorder_realization'],
        coords={
            'rho': data.rho,
            'h': data.h,
            'disorder_realization': data.disorder_realization,
            'ensemble': ["diagonal", "canonical", "micro"],
        }
    )
    return dataset

def micro_canonical_prediction(data, realizations=False):
    eev = np.average(data.eev, axis=-1) #average over spins
    shots = realizations or len(data.disorder_realization)
    fields = len(data.h)
    rhos = len(data.rho)
    results = np.zeros((rhos, fields, shots), dtype=np.float32)
    E_0 = np.einsum("abcd,abcd->abc",data.eon, data.e_vals) #->(rho, disorder, h)
    E_0_square = np.einsum("abcd,abcd->abc",data.eon, data.e_vals**2) #->(rho, disorder, h)
    E_0_var = E_0_square - E_0**2 #->(rho, disorder, h)
    for i in range(rhos):
        for j in range(fields):
            for k in range(shots):
                # use energy variance as window for microcanonical ensemble
                ensemble_occupation = sim.micro_ensemble(data.e_vals[i,k,j], E_0[i,k,j], delta_E=E_0_var[i,k,j])
                results[i,j,k] += np.dot(eev[i,k,j], ensemble_occupation)
    return results

def diagonal_prediction(data, realizations=False):
    shots = realizations or len(data.disorder_realization)
    results = np.einsum("abcde, abcd -> acb", data.eev[:,:shots], data.eon[:,:shots])
    results /= len(data.spin)
    return results

def canonical_prediction(data, realizations=False):
    eev = np.average(data.eev, axis=-1) #average over spins
    shots = realizations or len(data.disorder_realization)
    fields = len(data.h)
    rhos = len(data.rho)
    results = np.zeros((rhos, fields, shots), dtype=np.float32)
    E_0 = np.einsum("abcd,abcd->abc",data.eon, data.e_vals) #->(rho, disorder, h)
    simlib.log(f"ToDo: {rhos} rhos, {fields} field values and {shots} disorder realizations")
    for i in range(rhos):
        simlib.log(f"Starting rho #{i}\nField: ")
        for j in range(fields):
            simlib.log(j, end="\t")
            for k in range(shots):
                ensemble_occupation = sim.canonical_ensemble(data.e_vals[i,k,j], E_0[i,k,j])
                results[i,j,k] += np.dot(eev[i,k,j], ensemble_occupation)
    return results

## core function
def compute_ensemble_predictions(data, realizations=False):
    results = empty_result_set(data)

    results.loc["micro"]     = micro_canonical_prediction(data, realizations)
    results.loc["diagonal"]  = diagonal_prediction(data, realizations)
    results.loc["canonical"] = canonical_prediction(data, realizations)
    return results

## save/load functions
## abstracted to enable future optimization
def save_result(data, path, *params, **kwargs):
    "Either provide a full path to a .nc file or a directory and geometry and N in params"
    if params:
        path = simlib.result_data_path(path, *params, **kwargs)
    Path(path).parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)

def load_result(path, *params, **kwargs):
    if params:
        path = simlib.result_data_path(path, *params, **kwargs)
    return xr.load_dataarray(path)

## glue functions together
def main(path, force, realizations, geometry, dim, alpha, n_spins):
    "Load, compute and save results"
    save_path = simlib.result_data_path(path, geometry, dim, alpha, n_spins)
    if not force and save_path.exists():
        # check amount of realizations? field values?
        simlib.log(f"Results with params: geometry={geometry}, dim={dim}, N={n_spins}, alpha={alpha} already exist. Skipping. Use --force to overwrite.")
        exit()
    sim_data = edlib.load_data(path, geometry, dim, alpha, n_spins)
    result = compute_ensemble_predictions(sim_data, realizations)
    save_result(result, save_path)


if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser(description='Generate predictions for the mean x magentization.')
    parser.add_argument("-p", "--path", type=Path, default=Path.cwd(), help="Data directory. Results will be saved to 'results' subdirectory.")
    parser.add_argument("-d", "--dimensions", metavar="d", type=int, help="Number of spatial dimensions (1,2,3 are supported)", default=3)
    parser.add_argument("-a", "--alpha", metavar="alpha", type=int, help="coefficient of the vdW interactions", default=3)
    parser.add_argument("-r", "--realizations", metavar="n", type=int, help="Limit number of disorder samples.", default=False)
    parser.add_argument("-F", "--force", action="store_true", help="Force overwriting of existing data.")
    parser.add_argument("geometry", type=str, help="Geometry sampled from", choices=simlib.SAMPLING_GEOMETRIES)
    parser.add_argument("spins", type=int, help="Number of spins")
    args = parser.parse_args()

    main(args.path, args.force, args.realizations, args.geometry, args.dimensions, args.alpha, args.spins)


