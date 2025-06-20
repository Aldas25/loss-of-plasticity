#!/bin/sh

#SBATCH --partition=compute-p1
#SBATCH --time=02:00:00
#SBATCH --ntasks=2
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500MB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=/home/alenksas/slurm_outputs/slowly_regr_bp.%j.out
#SBATCH --error=/home/alenksas/slurm_outputs/slowly_regr_bp.%j.err

# Assuming this was done: ./prepare_data.sh
# and data is now ready

module load 2023r1
module load miniconda3
conda activate lop

# Launch tasks in parallel using "srun ... &"

# BP last
parent_dir="/scratch/alenksas/loss-of-plasticity/lop/slowly_changing_regression"

echo "Running job for BP 14.json"
srun --ntasks=1 ./slowly_regression_one_run.sh bp_temp_cfg 14.json &

# CBP last
echo "Running job for CBP 14.json"
srun --ntasks=1 ./slowly_regression_one_run.sh cbp_temp_cfg 14.json &

# Wait for all background tasks to complete
wait
