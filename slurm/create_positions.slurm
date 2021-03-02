#!/bin/sh 
# ########## Begin Slurm header ########## 
#SBATCH --nodes=1 
#SBATCH --ntasks-per-node=1
#SBATCH --time=01:00:00 
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

echo "[$(date +"%T")] Start!"
for n in {6..20}
do
    for d in {1..3}
    do
        echo "[$(date +"%T")] N=$n d=$d" 
        python3 scripts/positions.py --path $(ws_find cusp) -r 400 -d $d box-pbc $n
    done
done
echo "[$(date +"%T")] DONE!"