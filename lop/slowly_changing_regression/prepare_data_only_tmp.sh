#!/bin/bash

# Based on the README.md file in this folder.

# ALERT (you may waste your whole time if you forget this!)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!  You also need to change this number in the config (json) files that you use. !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

INDEX=$1

echo "Starting the script to prepare the data. INDEX=$INDEX"

parent_dir=$(dirname "$(realpath "$0")")
echo "Parent directory: $parent_dir"
cd "$parent_dir" || exit 1

#tmp_dir=/tmp/alenksas

# Remove previous data and create new directories
#rm -rf data utils_saved env_temp_cfg cbp_temp_cfg bp_temp_cfg
#mkdir -p env_temp_cfg cbp_temp_cfg bp_temp_cfg

# Create temporary configuration files in env_temp_cfg
#python3 multi_param_expr.py -c cfg/prob.json 

# Create data for each run
#for f in env_temp_cfg/*; do
#    echo "generating outputs with f = $f"
#    python3 slowly_changing_regression.py -c "$f"
#done

env_conf_file=env_temp_cfg/$INDEX.json
echo "generating outputs with $env_conf_file"
python3 slowly_changing_regression.py -c $env_conf_file


#for c_f in "cfg/sgd/cbp/relu.json" "cfg/sgd/bp/relu.json"; do
#	echo "Preparing data for experiments, config file: $c_f"

	# Create temporary configuration files in corresponding temp_cfg
#	python3 multi_param_expr.py -c $c_f
#done



echo "Done"
