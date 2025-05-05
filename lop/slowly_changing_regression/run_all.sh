#!/bin/bash

# Based on the README.md file in this folder.

# ALERT (you may waste your whole time if you forget this!)
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!
# !!!  You also need to change this number in the config (json) files that you use. !!!
# !!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!!

echo "Starting the script..."

# source ~/envs/lop/bin/activate

# Remove previous data and create new directories
rm -rf data env_temp_cfg temp_cfg utils_saved
mkdir -p env_temp_cfg temp_cfg

# Create temporary configuration files in env_temp_cfg
python3 multi_param_expr.py -c cfg/prob.json 

# Create data for each run
for f in env_temp_cfg/*; do
    python3 slowly_changing_regression.py -c "$f"
done

###
### CBP
###

echo "Starting the BP experiments..."

# Clear temp_cfg directory
rm -rf temp_cfg/*

# Create temporary configuration files in temp_cfg for the BP with Relu
python3 multi_param_expr.py -c cfg/sgd/bp/relu.json 

# Run the experiment for each configuration file
for f in temp_cfg/*; do
    echo "Running experiment with configuration file: $f"
    python3 expr.py -c "$f"
done

echo "Finished running the BP experiments."

###
### BP with Relu
###

# # Clear temp_cfg directory
# rm -rf temp_cfg/*

# # Create temporary configuration files in temp_cfg for the BP with Relu
# python3 multi_param_expr.py -c cfg/sgd/bp/relu.json 

# # Run the experiment for each configuration file
# for f in temp_cfg/*; do
#     python3 expr.py -c "$f"
# done

# # Generate the plots
# cd plots
# python3 online_performance.py -c ../cfg/sgd/bp/relu.json 
# cd ..