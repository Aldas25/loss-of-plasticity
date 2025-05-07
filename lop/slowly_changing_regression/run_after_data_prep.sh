#!/bin/bash

# Based on the README.md file in this folder.

# ALERT (you may waste your whole time if you forget this!)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!  You also need to change this number in the config (json) files that you use. !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

echo "Starting ..."

for config_dir in "cbp_temp_cfg" "bp_temp_cfg" "snp_temp_cfg"; do
    echo "Running for config files in the dir $config_dir"

    # Run the experiment for each configuration file
    for f in $config_dir/*; do
        echo "  Running experiment with configuration file: $f"
        python3 expr.py -c "$f"
    done

done


echo "Finished running the BP experiments."
