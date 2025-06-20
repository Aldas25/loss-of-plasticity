#!/bin/bash

CONFIG_DIR=$1
INDEX=$2
RUN_ID=$3

parent_dir="/home/nfs/alenksas/loss-of-plasticity/lop/permuted_mnist"
cd $parent_dir || exit 1

config_file="$parent_dir/${CONFIG_DIR}/${INDEX}.json"
echo "Processing $config_file"

echo "Prepare data script (load MNIST)"
python3 load_mnist.py $RUN_ID

echo "Running experiment (python3 expr.py) with $config_file"
python3 online_expr.py $RUN_ID -c $config_file 
