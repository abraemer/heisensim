import numpy as np
import xarray as xr
from multiprocessing import Pool

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

def compute(position_data, geometry, realizations, field_values, interaction, int_type, rhos=None):
    "Main computation routine"
    N = len(position_data.particle)
    dim = len(position_data.xyz)
    rhos = position_data.rho if rhos is None else position_data.rho[rhos]
    rhos = np.sort(np.asarray(rhos))
    simulation_results = empty_result_set(rhos, realizations, N, field_values)

    simlib.log(f"ToDo: {rhos}")
    simlib.log("with", realizations, "realizations and", len(field_values), "field values")
    for rho in rhos:
        geom = simlib.SAMPLING_GENERATORS[geometry](N=N, dim=dim, rho=float(rho))
        simlib.log(f"rho = {rho}")
        for i in range(realizations):
            simlib.log(f"{i:03d}/{realizations:03d}")
            model = sim.SpinModelSym(int_mat=interaction.get_interaction(geom, position_data.loc[rho, i]), int_type=int_type)
            normed_field_values = field_values * model.J_mean
            eev, eon, evals = compute_core(model, normed_field_values)
            simulation_results.e_vals.loc[rho, i] = evals
            simulation_results.eev.loc[rho, i] = eev
            simulation_results.eon.loc[rho, i] = eon

            # H_int = model.hamiltonian()
            # psi_0 = model.product_state()
            # # magn = 1 / N * sum(model.get_op_list(sim.sx))
            # # J_median = np.median(model.int_mat.sum(axis=0))

            # for h in field_values:
            #     H = H_int + model.hamiltonian_field(hx=h)
            #     e_vals, e_states = np.linalg.eigh(H.toarray())
            #     for spin, op in enumerate(model.get_op_list(sim.sx)):
            #         eev = sim.expect(op, e_states)
            #         simulation_results.eev.loc[rho, i, h, :, spin] = eev

            #     simulation_results.e_vals.loc[rho, i, h] = e_vals
            #     simulation_results.eon.loc[rho, i, h] = np.abs(psi_0 @ e_states)**2

    return simulation_results

def compute_parallel(position_data, geometry, realizations, field_values, interaction, int_type, rhos=None, processes=12):
    "Main computation routine using T processes"
    N = len(position_data.particle)
    dim = len(position_data.xyz)
    rhos = position_data.rho if rhos is None else position_data.rho[rhos]
    rhos = np.sort(np.asarray(rhos))
    simulation_results = empty_result_set(rhos, realizations, N, field_values)

    simlib.log(f"ToDo: {rhos}")
    simlib.log("with", realizations, "realizations and", len(field_values), "field values using", processes, "processes")
    with Pool(processes=processes) as pool:
        tasks = [[None]*realizations for _ in range(len(rhos))]
        for j, rho in enumerate(rhos):
            geom = simlib.SAMPLING_GENERATORS[geometry](N=N, dim=dim, rho=float(rho))
            for i in range(realizations):
                model = sim.SpinModelSym(int_mat=interaction.get_interaction(geom, position_data.loc[rho, i]), int_type=int_type)
                normed_field_values = field_values * model.J_mean
                tasks[j][i] = pool.apply_async(compute_core_process, args=(model, normed_field_values, f"rho #{j} - {i:03d}/{realizations:03d}"))
        simlib.log("Everthing started!")
        pool.close()
        pool.join()
        for j in range(len(rhos)):
            for i in range(realizations):
                eev, eon, evals = tasks[j][i].get()
                simulation_results.e_vals[j, i] = evals
                simulation_results.eev[j, i] = eev
                simulation_results.eon[j, i] = eon
    return simulation_results

def compute_core_process(model, field_values, name):
    simlib.log(name, "started!")
    res = compute_core(model, field_values)
    simlib.log(name, "finished!")
    return res

def compute_core(model, field_values):
    N = model.N
    dim = 2**(N-1)
    result_eev   = np.zeros((len(field_values), dim, N), dtype=np.float64)
    result_eon   = np.zeros((len(field_values), dim),    dtype=np.float64)
    result_evals = np.zeros((len(field_values), dim),    dtype=np.float64)

    spin_ops = model.get_op_list(sim.sx)
    field = sum(spin_ops)
    H_int = model.hamiltonian()
    psi_0 = model.product_state()
    for i, h in enumerate(field_values):
        H = H_int + h*field#model.hamiltonian_field(hx=h)
        e_vals, e_states = np.linalg.eigh(H.toarray())
        for spin, op in enumerate(spin_ops):
            result_eev[i, :, spin] = sim.expect(op, e_states)
        result_evals[i] = e_vals
        result_eon[i] = np.abs(psi_0 @ e_states)**2
    return result_eev, result_eon, result_evals



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
def main(path, force, realizations, geometry, dim, alpha, n_spins, field_values, rhos):
    "Load, compute and save results"
    save_path = simlib.ed_data_path(path, geometry, dim, alpha, n_spins)
    if not force and save_path.exists():
        # check amount of realizations? field values?
        simlib.log(f"Results with params: geometry={geometry}, dim={dim}, N={n_spins}, alpha={alpha} already exist. Skipping. Use --force to overwrite.")
        exit()
    position_data = poslib.load_positions(path, geometry, dim, n_spins)
    interaction = sim.PowerLaw(exponent=alpha, normalization=None)
    int_type = sim.XX()
    disorder_realizations = realizations or len(position_data.disorder_realization)
    field_values = np.sort(list(set(field_values)))
    rhos = np.sort(list(set(rhos))) if rhos else None
    result = compute_parallel(position_data, geometry, disorder_realizations, field_values, interaction, int_type, rhos)
    save_data(result, save_path)

## Take cmd args when used as script
if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(description='Calculate ensemble expectation values.')
    parser.add_argument("-p", "--path", type=Path, default=Path.cwd(), help="Data directory. Results will be saved to 'results' subdirectory.")
    parser.add_argument("-d", "--dimensions", metavar="d", type=int, help="Number of spatial dimensions (1,2,3 are supported)", default=3)
    parser.add_argument("-a", "--alpha", metavar="alpha", type=int, help="coefficient of the vdW interactions", default=3)
    parser.add_argument("-r", "--realizations", metavar="n", type=int, help="Limit number of disorder samples.", default=False)
    parser.add_argument("--rho", type=int, metavar="r", help="Override density", nargs="+")
    parser.add_argument("-F", "--force", action="store_true", help="Force overwriting of existing data.")
    parser.add_argument("geometry", type=str, help="Geometry sampled from", choices=simlib.SAMPLING_GEOMETRIES)
    parser.add_argument("spins", type=int, help="Number of spins")
    parser.add_argument("field", type=float, metavar="f", help="Effective external field, vary between -10 and 10", nargs="+")
    args = parser.parse_args()

    main(args.path, args.force, args.realizations, args.geometry, args.dimensions, args.alpha, args.spins, args.field, args.rho)
