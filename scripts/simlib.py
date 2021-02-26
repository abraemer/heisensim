# Some common functionality needed to run simulations
from pathlib import Path
import numpy as np

# add heisensim path to sys.path
import sys
sys.path.append(str(Path(__file__).parent.parent))
import heisensim as sim # pylint: disable=import-error

## Sampling related stuff
#TODO dimensionality
SAMPLING_GEOMETRIES = ["sphere", "box", "box-pbc"]#, "experiment" ?

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



# The OBC Volume coeffs should probably be a bit bigger to accommodate for the extra space outside
# Right now results won't really be comparable, but we will not need this (right now that is)
VOLUME_COEFFS = {"blockade":[1.0, np.pi, 4/3*np.pi], "sphere":[1.0, np.pi, 4/3*np.pi], "box":[1.0,1.0,1.0], "box-pbc":[1.0,1.0,1.0]}
SAMPLING_GENERATORS = {"sphere":sphere_sampler, "box":box_sampler, "box-pbc":box_pbc_sampler}


## Save paths
def position_data_path(prefix, geometry, dim, N):
    return Path(prefix) / "positions" / f"{geometry}_{dim}d_N_{N:02d}.nc"

def ed_data_path(prefix, geometry, dim, N): #r_bl, h
    return Path(prefix) / "results" / f"ed_{geometry}_{dim}d_N_{N:02d}.nc"