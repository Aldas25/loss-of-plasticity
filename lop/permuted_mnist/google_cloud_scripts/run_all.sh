#!/bin/bash

echo "Starting to run all..."

index_from1=0
index_to1=14

cd /home/aldas/loss-of-plasticity/lop/permuted_mnist/google_cloud_scripts || exit 1

LOG_DIR="/home/aldas/loss-of-plasticity/lop/permuted_mnist/google_cloud_scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE_BASE="$LOG_DIR/run_all_log"

# Run all the experiments in parallel, using 8 cores
nohup taskset -c 0 ./run_experiments_in_cfg_dir.sh bp_temp_cfg $index_from1 $index_to1 >> "$LOG_FILE_BASE-0.log" 2>&1 &
nohup taskset -c 1 ./run_experiments_in_cfg_dir.sh cbp_temp_cfg $index_from1 $index_to1 >> "$LOG_FILE_BASE-1.log" 2>&1 &
nohup taskset -c 2 ./run_experiments_in_cfg_dir.sh snp_temp_cfg $index_from1 $index_to1 >> "$LOG_FILE_BASE-2.log" 2>&1 &
nohup taskset -c 3 ./run_experiments_in_cfg_dir.sh l2_temp_cfg $index_from1 $index_to1 >> "$LOG_FILE_BASE-3.log" 2>&1 &

wait

echo "Finished running all."
