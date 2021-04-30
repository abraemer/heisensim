#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=00:40:00 
#SBATCH --mem=1gb 
#SBATCH --job-name=positions
#SBATCH --output="logs/pos-slurm-%j.out"
########### End Slurm header ##########
echo "Working Directory:          $PWD" 
echo "Running on host:            $HOSTNAME" 
echo "Job id:                     $SLURM_JOB_ID" 
echo "Job name:                   $SLURM_JOB_NAME" 
echo "Number of nodes allocated:  $SLURM_JOB_NUM_MODES" 
echo "Number of cores allocated:  $SLURM_NTASKS" 

# load modules
module load phys/qutip # also loads numpy, scipy, matplotlib
# pip install xarray # installed locally not needed each time

$SHOTS=$1
echo "Recreating position files with $SHOTS realizations"

echo "[$(date +"%T")] Start!"
for N in {6..20}
do
    echo "[$(date +"%T")] N=$N d=$d noisy-chain-pbc"
    python3 ./scripts/positions.py --path $(ws_find cusp) -r $SHOTS -F -d 1 noisy-chain-pbc $N
    for d in {1..3}
    do
        for geom in "box-pbc" #"sphere" "box"
        do
            echo "[$(date +"%T")] N=$N d=$d $geom" 
            python3 ./scripts/positions.py --path $(ws_find cusp) -r $SHOTS -F -d $d $geom $N
        done
    done
done
echo "[$(date +"%T")] DONE!"