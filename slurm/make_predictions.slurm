#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:05:00 
#SBATCH --mem=10gb 
#SBATCH --job-name=predictions
#SBATCH --cpus-per-task=1
########### End Slurm header ##########
echo "Working Directory:          $PWD" 
echo "Running on host:            $HOSTNAME" 
echo "Job id:                     $SLURM_JOB_ID" 
echo "Job name:                   $SLURM_JOB_NAME" 
echo "Number of nodes allocated:  $SLURM_JOB_NUM_MODES" 
echo "Number of cores allocated:  $SLURM_NTASKS" 

GEOM=$1
N=$2
D=$3
ALPHA=$4


# load modules
module load phys/qutip # also loads numpy, scipy, matplotlib
# pip install xarray # installed locally not needed each time

echo "[$(date +"%T")] $GEOM N=$N d=$D alpha=$ALPHA" 
python3 scripts/ensemble_predictions.py --path $(ws_find cusp) --alpha $ALPHA --dimensions $D $GEOM $N
echo "[$(date +"%T")] DONE"
