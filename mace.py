import h5py
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree

import heisensim as sim

parser = argparse.ArgumentParser(description='Calculate ensemble expectation values.')
parser.add_argument('--spin_number', type=int, default=10,
                    help='number of spins. Hilbert space has dimension 2**N')
parser.add_argument('--field', "-f", type=float, default=0.0,
                    help="External field, vary between -10 and 10")

args = parser.parse_args()
h = args.field
N = args.spin_number

path = Path.cwd()
with h5py.File(path / "positions" / 'positions.jld2', 'r') as file:
    positions = file["cusp_21_11_2020"]["pos"][:]
positions = np.array(positions)
tree = cKDTree(data=positions)
spins = positions.shape[0]

h_list = [h]
empty_array = np.zeros((spins, 1, 2 ** (N - 1)), dtype=np.float64)
empty_array_eev = np.zeros((spins, 1, 2 ** (N - 1), N), dtype=np.float64)
empty_array_E_0 = np.zeros((spins, 1), dtype=np.float64)
disorder_array = np.arange(spins)

simulation_results = xr.Dataset(
    {
        'e_vals': (['disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
        'eev': (['disorder_realization', 'h', 'eigen_state', 'spin'], empty_array_eev.copy()),
        'eon': (['disorder_realization', 'h', 'eigen_state'], empty_array.copy()),
        'E_0': (['disorder_realization', 'h'], empty_array_E_0.copy()),
        'delta_E_0': (['disorder_realization', 'h'], empty_array_E_0.copy())
    },
    coords={
        'disorder_realization': disorder_array,
        'h': [h],
        'eigen_state': np.arange(2 ** (N - 1)),
        'spin': np.arange(N)
    }
)

for i in disorder_array:
    pos_i = positions[i]
    pos_indices = tree.query_ball_point(pos_i, N + 1)
    print(pos_indices)
    pos = positions[pos_indices]
    model = sim.SpinModelSym.from_pos(pos, int_params=sim.DipoleCoupling(1, normalization='mean'),
                                      int_type=sim.XX())
    magn = 1 / N * sum(model.get_op_list(sim.sx))
    H_int = model.hamiltonian()
    J_median = np.median(model.int_mat.sum(axis=0))
    psi_0 = model.product_state()

    H = H_int + model.hamiltonian_field(hx=h)
    e_vals, e_states = np.linalg.eigh(H.toarray())
    for spin, op in enumerate(model.get_op_list(sim.sx)):
        eev = sim.expect(op, e_states)
        simulation_results.eev.loc[i, h, :, spin] = eev

    eon = psi_0 @ e_states
    E_0 = sim.expect(H, psi_0)
    delta_E_0 = np.sqrt(sim.expect(H @ H, psi_0) - E_0 ** 2)

    simulation_results.e_vals.loc[i, h] = e_vals
    simulation_results.eon.loc[i, h] = eon
    simulation_results.E_0.loc[i, h] = E_0
    simulation_results.delta_E_0.loc[i, h] = delta_E_0
    break

# save_path = path / "simulation_results"  #
save_path = Path("/scratch/users/jfranz/Heisenberg")
simulation_results.to_netcdf(
    save_path / "mace_N-{}_h-{}.nc".format(
        N, h)
)
