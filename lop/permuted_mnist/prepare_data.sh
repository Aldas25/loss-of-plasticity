#!/bin/bash

# Based on the README.md file in this folder.

# WARNING: Don't forget to change the config files before running this script.

parent_dir=$(dirname "$(realpath "$0")")
echo "Parent directory: $parent_dir"
cd "$parent_dir" || exit 1

# Remove previous data and create new directories
rm -rf data cbp_temp_cfg bp_temp_cfg snp_temp_cfg l2_temp_cfg utils_saved 
mkdir -p cbp_temp_cfg bp_temp_cfg snp_temp_cfg l2_temp_cfg data

echo "Loading MNIST data..."

# Download the data 
python3 load_mnist.py

echo "Data loaded. Creating temp cfg files..."

# Create temporary configuration files for each run.
echo "  ... BP temp cfg files..."
python3 multi_param_expr.py -c cfg/bp/std_net.json
echo "  ... CBP temp cfg files..."
python3 multi_param_expr.py -c cfg/cbp.json
echo "  ... S&P temp cfg files..."
python3 multi_param_expr.py -c cfg/snp.json
echo "  ... L2 temp cfg files..."
python3 multi_param_expr.py -c cfg/l2.json

echo "Finished."
