#!/bin/sh

echo "start"

module use /opt/insy/modulefiles
module load miniconda
conda activate lop

echo "end"

