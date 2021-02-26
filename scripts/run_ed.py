import numpy as np
import xarray as xr
from pathlib import Path

# add heisensim path to sys.path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import heisensim as sim # pylint: disable=import-error
import scripts.simlib as simlib # pylint: disable=import-error
import scripts.positions as poslib # pylint: disable=import-error


def empty_result_set(rho, realizations, system_size, field_values):
    "creates an empty result set"
    n_radii = len(rho)
    n_states = 2**(system_size-1) # 2**N/2 due to spin flip symmetry
    n_fields = len(field_values)

    empty_array_eev = np.zeros((n_radii, realizations, n_fields, n_states, system_size), dtype=np.float64)
    empty_array     = np.zeros((n_radii, realizations, n_fields, n_states), dtype=np.float64)
    # empty_array_E_0 = np.zeros((n_radii, realizations, n_fields), dtype=np.float64)

    simulation_results = xr.Dataset(
    {
        'eev':       (['rho', 'disorder_realization', 'h', 'eigen_state', 'spin'], empty_array_eev.copy()),
        'e_vals':    (['rho', 'disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
        'eon':       (['rho', 'disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
    #    'E_0':       (['rho', 'disorder_realization', 'h'], empty_array_E_0.copy()),
    #    'delta_E_0': (['rho', 'disorder_realization', 'h'], empty_array_E_0.copy())
    },
    coords={
        'rho': rho,
        'disorder_realization': np.arange(realizations),
        'h': field_values,
        'eigen_state': np.arange(n_states),
        'spin': np.arange(system_size)
    }
)
    return simulation_results

def compute(position_data, geometry, dim, realizations, field_values, alpha=3):
    "Main computation routine"
    N = len(position_data.particle)
    dim = len(position_data.xyz)
    simulation_results = empty_result_set(position_data.rho, realizations, N, field_values)
    interaction = sim.PowerLaw(exponent=alpha, normalization='mean')
    interaction = sim.DipoleCoupling(1, normalization='mean')

    print("ToDo:", np.array(position_data.rho))
    for rho in position_data.rho:
        geom = simlib.SAMPLING_GENERATORS[geometry](N=N, dim=dim, rho=float(rho))
        print("rho =", float(rho))
        for i in range(realizations):
            model = sim.SpinModelSym(int_mat=interaction.get_interaction(geom, position_data.loc[rho, i]), int_type=sim.XX())
            H_int = model.hamiltonian()
            psi_0 = model.product_state()
            # magn = 1 / N * sum(model.get_op_list(sim.sx))
            # J_median = np.median(model.int_mat.sum(axis=0))

            for h in field_values:
                H = H_int + model.hamiltonian_field(hx=h)
                e_vals, e_states = np.linalg.eigh(H.toarray())
                for spin, op in enumerate(model.get_op_list(sim.sx)):
                    eev = sim.expect(op, e_states)
                    simulation_results.eev.loc[rho, i, h, :, spin] = eev

                simulation_results.e_vals.loc[rho, i, h] = e_vals
                simulation_results.eon.loc[rho, i, h] = np.abs(psi_0 @ e_states)**2
                ## Do not need this data. It can be computed much cheaper from eon and e_vals
                # E_0 = sim.expect(H, psi_0)
                # delta_E_0 = np.sqrt(sim.expect(H @ H, psi_0) - E_0 ** 2)
                # simulation_results.E_0.loc[rho, i, h] = E_0
                # simulation_results.delta_E_0.loc[rho, i, h] = delta_E_0

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
def main(path, realizations, geometry, dim, n_spins, field_values):
    "Load, compute and save results"
    position_data = poslib.load_positions(path, geometry, dim, n_spins)
    disorder_realizations = realizations or len(position_data.disorder_realization)
    result = compute(position_data, geometry, dim, disorder_realizations, field_values)
    save_data(result, path, geometry, dim, n_spins)

## Take cmd args when used as script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calculate ensemble expectation values.')
    parser.add_argument("-p", "--path", type=Path, default=Path.cwd(), help="Data directory. Results will be saved to 'results' subdirectory.")
    parser.add_argument("-d", "--dimensions", metavar="d", type=int, help="Number of spatial dimensions (1,2,3 are supported)", default=3)
    parser.add_argument("-r", "--realizations", metavar="n", type=int, help="Limit number of disorder samples.", default=False)
    parser.add_argument("geometry", type=str, help="Geometry sampled from", choices=simlib.SAMPLING_GEOMETRIES)
    parser.add_argument("spins", type=int, help="Number of spins")
    parser.add_argument("field", type=float, metavar="f", help="External field, vary between -10 and 10", nargs="+")
    args = parser.parse_args()

    main(args.path, args.realizations, args.geometry, args.dimensions, args.spins, np.sort(list(set(args.field))))
