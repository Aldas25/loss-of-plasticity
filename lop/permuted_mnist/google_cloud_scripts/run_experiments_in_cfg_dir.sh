#!/bin/bash

echo "Starting ..."

cd /home/aldas/loss-of-plasticity/lop/permuted_mnist || exit 1
conda activate lop

cfg_dir=$1
index_from=$2
index_to=$3

echo "Running for config files in the dir $cfg_dir, indices from $index_from to $index_to."

log_dir="/home/aldas/loss-of-plasticity/lop/permuted_mnist/google_cloud_scripts/logs"
mkdir -p "$log_dir"

echo "Running for config files in the dir $cfg_dir"

# Run the experiment for each configuration file between the given indices (inclusive)
for (( i=${index_from}; i<=${index_to}; i++ )); do 
    f="$cfg_dir/$i.json"

    LOG_FILE="$log_dir/$cfg_dir-$i.log"

    echo "  Running experiment with configuration file: $f, will be logged to: $LOG_FILE"
    python3 online_expr.py -c "$f" >> "$LOG_FILE" 2>&1
done

echo "Finished for config files in the dir $cfg_dir, indices from $index_from to $index_to."
