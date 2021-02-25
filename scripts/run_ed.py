import numpy as np
import xarray as xr
from pathlib import Path

# add heisensim path to sys.path
import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import heisensim as sim # pylint: disable=import-error
import scripts.simlib as simlib # pylint: disable=import-error
import scripts.positions as poslib # pylint: disable=import-error


def empty_result_set(r_bl, realizations, system_size, field_values):
    "creates an empty result set"
    n_radii = len(r_bl)
    n_states = 2**(system_size-1) # 2**N/2 due to spin flip symmetry
    n_fields = len(field_values)

    empty_array     = np.zeros((n_radii, realizations, n_fields, n_states), dtype=np.float64)
    empty_array_eev = np.zeros((n_radii, realizations, n_fields, n_states, system_size), dtype=np.float64)
    empty_array_E_0 = np.zeros((n_radii, realizations, n_fields), dtype=np.float64)

    simulation_results = xr.Dataset(
    {
        'eev':       (['r_bl', 'disorder_realization', 'h', 'eigen_state', 'spin'], empty_array_eev.copy()),
        'e_vals':    (['r_bl', 'disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
        'eon':       (['r_bl', 'disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
    #    'E_0':       (['r_bl', 'disorder_realization', 'h'], empty_array_E_0.copy()),
    #    'delta_E_0': (['r_bl', 'disorder_realization', 'h'], empty_array_E_0.copy())
    },
    coords={
        'r_bl': r_bl,
        'disorder_realization': np.arange(realizations),
        'h': field_values,
        'eigen_state': np.arange(n_states),
        'spin': np.arange(system_size)
    }
)
    return simulation_results

def compute(position_data, realizations, field_values):
    "Main computation routine"
    simulation_results = empty_result_set(position_data.r_bl, realizations, len(position_data.particle), field_values)

    print("ToDo:", np.array(position_data.r_bl))
    for r_bl in position_data.r_bl:
        print("r_bl =", float(r_bl))
        for i in range(realizations):
            model = sim.SpinModelSym.from_pos(position_data.loc[r_bl, i], int_params=sim.DipoleCoupling(1, normalization='mean'), int_type=sim.XX())
            H_int = model.hamiltonian()
            psi_0 = model.product_state()
            # magn = 1 / N * sum(model.get_op_list(sim.sx))
            # J_median = np.median(model.int_mat.sum(axis=0))

            for h in field_values:
                H = H_int + model.hamiltonian_field(hx=h)
                e_vals, e_states = np.linalg.eigh(H.toarray())
                for spin, op in enumerate(model.get_op_list(sim.sx)):
                    eev = sim.expect(op, e_states)
                    simulation_results.eev.loc[r_bl, i, h, :, spin] = eev

                simulation_results.e_vals.loc[r_bl, i, h] = e_vals
                simulation_results.eon.loc[r_bl, i, h] = np.abs(psi_0 @ e_states)**2
                ## Do not need this data. It can be computed much cheaper from eon and e_vals
                # E_0 = sim.expect(H, psi_0)
                # delta_E_0 = np.sqrt(sim.expect(H @ H, psi_0) - E_0 ** 2)
                # simulation_results.E_0.loc[r_bl, i, h] = E_0
                # simulation_results.delta_E_0.loc[r_bl, i, h] = delta_E_0

    return simulation_results

## Save/Load
def save_data(data, path, *params):
    # TODO merge if compatible (e.g only different field values)
    path = Path(path)
    if params:
        path = simlib.ed_data_path(path, *params)
    path.parent.mkdir(parents=True, exist_ok=True)
    data.to_netcdf(path)
    return

def load_data(path, *params):
    path = Path(path)
    if params:
        path = simlib.ed_data_path(path, *params)
    return xr.open_dataset(path)

## Glue everything together
def main(path, realizations, geometry, n_spins, field_values):
    "Load, compute and save results"
    position_data = poslib.load_positions(path, geometry, n_spins)
    disorder_realizations = realizations or len(position_data.disorder_realization)
    result = compute(position_data, disorder_realizations, field_values)
    save_data(result, path, geometry, n_spins)

## Take cmd args when used as script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calculate ensemble expectation values.')
    parser.add_argument("-p", "--path", type=Path, default=Path.cwd(), help="Data directory. Results will be saved to 'results' subdirectory.")
    parser.add_argument("-r", "--realizations", metavar="n", type=int, help="Limit number of disorder samples.", default=False)
    parser.add_argument("geometry", type=str, help="Geometry sampled from", choices=simlib.SAMPLING_GEOMETRIES)
    parser.add_argument("spins", type=int, help="Number of spins")
    parser.add_argument("field", type=float, metavar="f", help="External field, vary between -10 and 10", nargs="+")
    args = parser.parse_args()

    main(args.path, args.realizations, args.geometry, args.spins, np.sort(list(set(args.field))))
    sys.exit()