# Some common functionality needed to run simulations
from pathlib import Path

# add heisensim path to sys.path
import sys, os.path
sys.path.append(os.path.dirname(os.path.dirname(os.path.realpath(__file__))))
import heisensim as sim # pylint: disable=import-error

## Sampling related stuff
#TODO dimensionality
SAMPLING_GEOMETRIES = ["sphere", "box"]#, "experiment" ?

## General idea:
## Scale the space by the amount of particles N s.t.
## total space is proportional to N * V1 / max_density
## where      V1     - volume of 1 particle at blockade radius 1.0
##       max_density - maximum packing density at given dimension
## This should make the disorder approximately independent of particle number and dimension
## s.t. it only depends on the blockade radius and should roughly be comparable across geometries.
##TODO can check different metrics for disorder to show this? spatial pdf f.e.?

## sphere geometry
def sphere_sampler(r_bl, N):
    return sim.Sphere(r_bl=r_bl, radius=SAMPLING_SCALING_FUNCTIONS["sphere"](N=N, r_bl=1), max_iter=int(1e5))

def radius_from_packing(N, r_bl=1, packing_density=0.74):
    # V = 
    return r_bl/2 * ((N/packing_density)**(1/3))

##box geometry
def box_sampler(r_bl, N):
    boxsize = SAMPLING_SCALING_FUNCTIONS["box"](N=N, r_bl=1)
    return sim.Box(length_x=boxsize, length_y=boxsize, length_z=boxsize, r_bl=r_bl, max_iter=int(1e5))

# we don't really need this. The scaling needs to be the same. Perhaps one could argue for a different coefficient.
#TODO think about a coefficient
def boxlength_from_packing(N, r_bl=1, packing_density=0.74):
    return (4*3.1415/3)**(1/3) * radius_from_packing(N, r_bl=1, packing_density=0.74)

SAMPLING_SCALING_FUNCTIONS = {"sphere":radius_from_packing, "box":boxlength_from_packing}
SAMPLING_GENERATORS = {"sphere":sphere_sampler, "box":box_sampler}


## Save paths
def position_data_path(prefix, geometry, N):
    return Path(prefix) / "positions" / f"{geometry}_positions_{N:02d}.nc"

def result_data_path(prefix, geometry, N, r_bl, h):
    return Path(prefix) / "results" / f"run_{geometry}_N-{N:02d}_rbl-{r_bl:.2f}_h-{h:.2f}.nc"
