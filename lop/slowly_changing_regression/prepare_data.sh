#!/bin/bash

# Based on the README.md file in this folder.

# ALERT (you may waste your whole time if you forget this!)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!  You also need to change this number in the config (json) files that you use. !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

echo "Starting the script to prepare the data."

parent_dir=$(dirname "$(realpath "$0")")
echo "Parent directory: $parent_dir"
cd "$parent_dir" || exit 1

# Remove previous data and create new directories
rm -rf data env_temp_cfg cbp_temp_cfg bp_temp_cfg snp_temp_cfg utils_saved 
mkdir -p env_temp_cfg cbp_temp_cfg bp_temp_cfg snp_temp_cfg

# Create temporary configuration files in env_temp_cfg
python3 multi_param_expr.py -c cfg/prob.json 

# Create data for each run
for f in env_temp_cfg/*; do
    python3 slowly_changing_regression.py -c "$f"
done

for config_file in "cfg/sgd/cbp/relu.json" "cfg/sgd/bp/relu.json" "cfg/sgd/shrink-and-perturb/snp.json"; do

    echo "Preparing data for experiments, config file: $config_file"

    # Create temporary configuration files in corresponding temp_cfg
    python3 multi_param_expr.py -c $config_file
done


echo "Done"