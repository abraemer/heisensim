#!/bin/bash
sbatch --output="logs/run-sa-N_$1-1d-alpha_6-slurm-%j.out" slurm/scaling_analysis.slurm $1