#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00 
#SBATCH --mem=1gb 
#SBATCH --job-name=test_script
#SBATCH --cpus-per-task=24
#SBATCH --output="logs/run-ed-slurm-%j.out"
########### End Slurm header ##########
echo "Working Directory:          $PWD" 
echo "Running on host:            $HOSTNAME" 
echo "Job id:                     $SLURM_JOB_ID" 
echo "Job name:                   $SLURM_JOB_NAME" 
echo "Number of nodes allocated:  $SLURM_JOB_NUM_MODES" 
echo "Number of cores allocated:  $SLURM_NTASKS" 

N=$1
D=$2
ALPHA=$3
GEOM=box-pbc

# load modules
module load phys/qutip # also loads numpy, scipy, matplotlib
# pip install xarray # installed locally not needed each time
export OMP_NUM_THREADS=24
export MKL_NUM_THREADS=24

echo "[$(date +"%T")] $GEOM N=$N d=$D alpha=$ALPHA" 
python3 scripts/run_ed.py --path $(ws_find cusp) -a $ALPHA -d $d $GEOM $n $(seq -1.0 0.04 0.75) $(seq -0.3 0.002 0.1)
echo "[$(date +"%T")] DONE!"
echo "Queuing evaluation"
sbatch slurm/make_predictions.slurm $GEOM $N $D $ALPHA