#!/bin/sh

#SBATCH --partition=compute-p1
#SBATCH --time=03:58:55
#SBATCH --ntasks=7
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=500MB
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=/home/alenksas/slurm_outputs/slowly_regr_cbp.%A.%a.out
#SBATCH --error=/home/alenksas/slurm_outputs/slowly_regr_cbp.%A.%a.err
#SBATCH --array=0-3

# Assuming this was done: ./prepare_data.sh
# and data is now ready

module load 2023r1
module load miniconda3
conda activate lop

TASKS_IN_THIS_BATCH=7
TEMP_CFG_DIR="cbp_temp_cfg"
# possible dirs: cbp_temp_cfg, bp_temp_cfg, snp_temp_cfg, l2_temp_cfg

BATCH_START=$((SLURM_ARRAY_TASK_ID * TASKS_IN_THIS_BATCH))
echo "Array task ${SLURM_ARRAY_TASK_ID} processing indices ${BATCH_START} to $((BATCH_START + TASKS_IN_THIS_BATCH - 1))"

# Launch tasks in parallel using "srun ... &"
for (( i=0; i<${TASKS_IN_THIS_BATCH}; i++ ))
do
  INDEX=$((BATCH_START + i))
  parent_dir="/scratch/alenksas/loss-of-plasticity/lop/slowly_changing_regression"

  config_file="$parent_dir/${TEMP_CFG_DIR}/${INDEX}.json"

  # Check if config file exists before running the task
  if [ -f "$config_file" ]; then
    echo "Config file $config_file exists. Running job for index $INDEX."
    srun --ntasks=1 ./slowly_regression_one_run.sh ${TEMP_CFG_DIR} ${INDEX} &
  else
    echo "Config file $config_file does not exist. Skipping index $INDEX."
  fi


done

# Wait for all background tasks to complete
wait
