#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=24:00:00 
#SBATCH --mem=64gb 
#SBATCH --job-name=SA
#SBATCH --cpus-per-task=48
########### End Slurm header ##########
echo "Working Directory:          $PWD" 
echo "Running on host:            $HOSTNAME" 
echo "Job id:                     $SLURM_JOB_ID" 
echo "Job name:                   $SLURM_JOB_NAME" 
echo "Number of nodes allocated:  $SLURM_JOB_NUM_MODES" 
echo "Number of cores allocated:  $SLURM_NTASKS" 

N=$1
D=1
ALPHA=6
GEOM=box-pbc
PROCESSES=12

# load modules
module load phys/qutip # also loads numpy, scipy, matplotlib
# pip install xarray # installed locally not needed each time
export OMP_NUM_THREADS=8
export MKL_NUM_THREADS=8

echo "[$(date +"%T")] $GEOM N=$N d=$D alpha=$ALPHA" 
python3 scripts/run_ed.py --path "$(ws_find cusp)/scaling" --alpha $ALPHA --processes $PROCESSES --rho 0 10 19 -r 400 --dimensions $D $GEOM $N -0.3 -0.15 -0.1 0.1
echo "[$(date +"%T")] DONE!"
echo "Queuing evaluation"
sbatch --output "logs/sa-prediction-N_$N-${D}d-alpha_$ALPHA-slurm-%j.out" slurm/make_predictions.slurm $GEOM $N $D $ALPHA
