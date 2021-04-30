# Some common functionality needed to run simulations
from pathlib import Path
from datetime import datetime as dt
import numpy as np

# add heisensim path to sys.path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import heisensim as sim # pylint: disable=import-error


## logging
VERBOSE = True
def log(*message, **kwargs):
    if VERBOSE:
        print(f"{dt.now().strftime('[%Y-%m-%d %H:%M:%S]')}", *message, **kwargs)
        sys.stdout.flush()


## Sampling related stuff
SAMPLING_GEOMETRIES = ["sphere", "box", "box-pbc", "noisy-chain-pbc"]#, "experiment" ?

## General idea:
## set r_bl=1 and scale volume via the density rho
## rho is defined as the ratio (blockaded volume) / (total volume)
## total volume = VOLUME_FACTOR[geom][d]*L**d
## blockaded volume = N*VOLUME_FACTOR[sphere][d]*1**d
def length_from_density(rho, geometry, N, dim=3):
    return (1/rho * N * VOLUME_COEFFS["blockade"][dim-1] / VOLUME_COEFFS[geometry][dim-1])**(1/dim)

## sphere geometry
def sphere_sampler(rho, N, dim=3):
    r = length_from_density(rho, "sphere", N, dim)
    return sim.Sphere(radius=r, dim=dim)

##box geometry
def box_sampler(rho, N, dim=3):
    size = length_from_density(rho, "box", N, dim)
    return sim.Box(lengths=np.ones(dim)*size)

## box geometry with periodic boundary conditions
def box_pbc_sampler(rho, N, dim=3):
    size = length_from_density(rho, "box-pbc", N, dim)
    return sim.BoxPBC(lengths=np.ones(dim)*size)

## noisy chain geometry with periodic boundary conditions
def noisy_chain_pbc_sampler(rho, N, dim=1):
    assert dim == 1
    # know that rho = 2/spacing, since r_bl = 1, and V = N*spacing
    spacing = 2/rho
    sigma = 1.5*(spacing-1) # this ensures ~90% success rate when generating positions -> see demo/rho_scaling.ipynb
    return sim.NoisyChain(N=N, spacing=spacing, sigma=sigma)


# The OBC Volume coeffs should probably be a bit bigger to accommodate for the extra space outside
# Right now results won't really be comparable, but we will not need this (right now that is)
VOLUME_COEFFS = {"blockade":[2.0, np.pi, 4/3*np.pi], "sphere":[2.0, np.pi, 4/3*np.pi], "box":[1.0,1.0,1.0], "box-pbc":[1.0,1.0,1.0]}
SAMPLING_GENERATORS = {"sphere":sphere_sampler, "box":box_sampler, "box-pbc":box_pbc_sampler, "noisy-chain-pbc":noisy_chain_pbc_sampler}


## Save paths
def position_data_path(prefix, geometry, dim, N):
    return Path(prefix) / "positions" / f"{geometry}_{dim}d_N_{N:02d}.nc"

def ed_data_path(prefix, geometry, dim, alpha, N):
    if int(alpha) == alpha:
        alpha = int(alpha)
    return Path(prefix) / "data" / f"ed_{geometry}_{dim}d_alpha_{alpha}_N_{N:02d}.nc"

def result_data_path(prefix, geometry, dim, alpha, N):
    if int(alpha) == alpha:
        alpha = int(alpha)
    return Path(prefix) / "results" / f"{geometry}_{dim}d_alpha_{alpha}_N_{N:02d}.nc"