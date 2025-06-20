#!/bin/bash

CONFIG_DIR=$1
INDEX=$2
NUM_RUNS=$3
RUN_ID=$4

parent_dir="/home/nfs/alenksas/loss-of-plasticity/lop/slowly_changing_regression"
cd $parent_dir || exit 1

config_file="$parent_dir/${CONFIG_DIR}/${INDEX}.json"
echo "Processing $config_file"

echo "Prepare data script"
MOD=$(( INDEX % NUM_RUNS ))
echo "   data will be taken from file with index $MOD"
./prepare_data_only_tmp.sh $MOD $RUN_ID

echo "Running experiment (python3 expr.py) with $config_file"
python3 expr.py $RUN_ID -c $config_file 
