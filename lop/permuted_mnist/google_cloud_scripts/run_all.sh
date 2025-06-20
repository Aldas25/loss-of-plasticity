#!/bin/bash

echo "Starting to run all..."

# index_from1=0
# index_to1=14

cd /home/aldas/loss-of-plasticity/lop/permuted_mnist/google_cloud_scripts || exit 1

LOG_DIR="/home/aldas/loss-of-plasticity/lop/permuted_mnist/google_cloud_scripts/logs"
mkdir -p "$LOG_DIR"
LOG_FILE_BASE="$LOG_DIR/run_all_log"

cfg_dir="cbp_snp_temp_cfg"

# Run all the experiments in parallel, using 8 cores
taskset -c 0 ./run_experiments_in_cfg_dir.sh $cfg_dir "0 4 8 12 16 20 24 28" >> "$LOG_FILE_BASE-0.log" 2>&1 &
taskset -c 1 ./run_experiments_in_cfg_dir.sh $cfg_dir "1 5 9 13 17 21 25 29" >> "$LOG_FILE_BASE-1.log" 2>&1 &
taskset -c 2 ./run_experiments_in_cfg_dir.sh $cfg_dir "2 6 10 14 18 22 26" >> "$LOG_FILE_BASE-2.log" 2>&1 &
taskset -c 3 ./run_experiments_in_cfg_dir.sh $cfg_dir "3 7 11 15 19 23 27" >> "$LOG_FILE_BASE-3.log" 2>&1 &

wait

echo "Finished running all."
