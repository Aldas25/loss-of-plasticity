#!/bin/bash

# This script takes the index as an argument
CONFIG_DIR=$1
INDEX=$2

parent_dir="/scratch/alenksas/loss-of-plasticity/lop/permuted_mnist"
cd $parent_dir || exit 1

config_file="$parent_dir/${CONFIG_DIR}/${INDEX}.json"
echo "Processing $config_file"
python3 online_expr.py -c $config_file 