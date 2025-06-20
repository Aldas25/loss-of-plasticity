#!/bin/sh

#SBATCH --partition=compute-p2
#SBATCH --time=03:55:00
#SBATCH --ntasks=3
#SBATCH --cpus-per-task=1
#SBATCH --mem-per-cpu=4000M
#SBATCH --account=education-eemcs-courses-cse3000
#SBATCH --output=/home/alenksas/slurm_outputs/perform_permuted_mnist.%A.%a.out
#SBATCH --error=/home/alenksas/slurm_outputs/perform_permuted_mnist.%A.%a.err
#SBATCH --array=0-0

# Assuming this was done: (in permuted_mnist folder) ./prepare_data.sh
# and data is now ready

module load 2023r1
module load miniconda3
conda activate lop

mkdir -p /home/alenksas/slurm_outputs

TASKS_IN_THIS_BATCH=1

BATCH_START=$((SLURM_ARRAY_TASK_ID * TASKS_IN_THIS_BATCH))
echo "Array task ${SLURM_ARRAY_TASK_ID} processing indices ${BATCH_START} to $((BATCH_START + TASKS_IN_THIS_BATCH - 1))"

# Launch tasks in parallel using "srun ... &"
for (( i=0; i<${TASKS_IN_THIS_BATCH}; i++ ))
do
  INDEX=$((BATCH_START + i))
  parent_dir="/scratch/alenksas/loss-of-plasticity/lop/permuted_mnist"

  for config_dir in "bp_temp_cfg"; do
  # for config_dir in "cbp_temp_cfg" "bp_temp_cfg" "snp_temp_cfg"; do
    config_file="$parent_dir/${config_dir}/${INDEX}.json"

    # Check if config file exists before running the task
    if [ -f "$config_file" ]; then
      echo "Config file $config_file exists. Running job for index $INDEX."
      srun --ntasks=1 ./permuted_mnist_one_run.sh ${config_dir} ${INDEX} &
    else
      echo "Config file $config_file does not exist. Skipping index $INDEX."
    fi

  done

done

# Wait for all background tasks to complete
wait
