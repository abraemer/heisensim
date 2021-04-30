#!/bin/bash
if [! $# -eq 5]
then
    echo "Usage: start_run.sh N D ALPHA GEOM"
    exit
fi
sbatch --output="logs/run-ed-N_$1-$2d-alpha_$3-slurm-%j.out" slurm/run_ed.slurm $1 $2 $3 $4