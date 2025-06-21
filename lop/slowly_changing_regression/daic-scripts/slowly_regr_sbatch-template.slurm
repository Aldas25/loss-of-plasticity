#!/bin/sh

#SBATCH --partition=insy,ewi-insy,general
#SBATCH --qos=short
#SBATCH --account=ewi-insy-sdm
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=2
#SBATCH --mem-per-cpu=1024MB
#SBATCH --time=02:00:00
# also add: #SBATCH --output=/home/nfs/alenksas/slurm_outputs/slowly_regr_cbp.%j.out
# also add: #SBATCH --error=/home/nfs/alenksas/slurm_outputs/slowly_regr_cbp.%j.err
# module use /opt/insy/modulefiles
# module load miniconda
# conda activate lop
# use: srun ./slowly_regr_one_run.sh bp_temp_cfg 0

