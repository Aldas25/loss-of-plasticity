#!/bin/bash

# Based on the README.md file in this folder.

echo "Starting ..."

for config_dir in "cbp_temp_cfg" "bp_temp_cfg" "snp_temp_cfg"; do
    echo "Running for config files in the dir $config_dir"

    # Run the experiment for each configuration file
    for f in $config_dir/*; do
        echo "  Running experiment with configuration file: $f"
        python3 online_expr.py -c "$f"
    done

done


echo "Finished."
