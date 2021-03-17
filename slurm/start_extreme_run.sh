#!/bin/bash
sbatch --output="logs/extreme-run-ed-N_$1-$2d-alpha_$3-slurm-%j.out" slurm/extreme_run_ed.slurm $1 $2 $3