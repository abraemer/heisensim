import h5py
from pathlib import Path
import argparse
import numpy as np
import xarray as xr
from pathlib import Path
from scipy.spatial import cKDTree
import pandas as pd

import heisensim as sim

parser = argparse.ArgumentParser(description='Calculate ensemble expectation values.')
parser.add_argument('--cluster_size', type=int, default=10,
                    help='number of spins. Hilbert space has dimension 2**N')
parser.add_argument('--field', "-f", type=float, default=-0.1,
                    help="External field, vary between -10 and 10")

args = parser.parse_args()
h = args.field
cluster_size = args.cluster_size

path = Path.cwd()
with h5py.File(path / "positions" / 'positions.jld2', 'r') as file:
    positions = file["cusp_21_11_2020"]["pos"][:]
positions = np.array(positions)
tree = cKDTree(data=positions)
spins = positions.shape[0]

results = pd.DataFrame(columns=["E_0", "delta_E_0", "diag", "micro", "canonical"])
for spin in range(spins):
    spin=20
    pos_i = positions[spin]
    _, pos_indices = tree.query(pos_i, cluster_size)
    pos = positions[pos_indices]
    model = sim.SpinModelSym.from_pos(pos, int_params=sim.DipoleCoupling(1200, normalization=None),
                                      int_type=sim.XX())
    magn = 1 / cluster_size * sum(model.get_op_list(sim.sx))
    H_int = model.hamiltonian()
    J_median = np.median(model.int_mat.sum(axis=0))
    psi_0 = model.product_state()

    H = H_int + model.hamiltonian_field(hx=h)
    e_vals, e_states = np.linalg.eigh(H.toarray())
    eon = psi_0 @ e_states

    E_0 = sim.expect(H, psi_0)
    delta_E_0 = np.sqrt(sim.expect(H @ H, psi_0) - E_0 ** 2)
    op = model.single_spin_op(sim.sx, 0)
    eev = sim.expect(op, e_states)
    diag = eev @ sim.diagonal_ensemble(eon)
    micro = eev @ sim.micro_ensemble(e_vals, E_0, delta_E=0.05)
    canonical = eev @ sim.canonical_ensemble(e_vals, E_0)

    results.loc[spin] = ([E_0, delta_E_0, diag, micro, canonical])

# save_path = path / "simulation_results"  #
save_path = Path("/scratch/users/jfranz/Heisenberg")
results.to_csv(
    save_path / "mace_cluster-{}_h-{}.csv".format(
        cluster_size, h)
)
