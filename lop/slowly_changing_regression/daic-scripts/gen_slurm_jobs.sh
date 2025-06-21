#!/bin/bash

output_dir=slurm_jobs

mkdir -p $output_dir

algo=$1
index_from=$2
index_to=$3
num_runs=$4

echo "doing for $algo indices: [$index_from, $index_to]; num_runs = $num_runs"

for ((i=$index_from; i<=$index_to; i++)); do
	result_file=$output_dir/$algo-$i.slurm
	temp_cfg_suf="_temp_cfg"
	cp slowly_regr_sbatch-template.slurm $result_file
	text="#SBATCH --output=/home/nfs/alenksas/slurm_outputs/slowly_regr_$algo-$i.%j.out"
	text="$text\n#SBATCH --error=/home/nfs/alenksas/slurm_outputs/slowly_regr_$algo-$i.%j.err"
	text="$text\nmodule use /opt/insy/modulefiles"
	text="$text\nmodule load miniconda"
	text="$text\nconda activate lop"
	text="$text\nsrun ./slowly_regr_one_run.sh $algo$temp_cfg_suf $i $num_runs $algo-$i"
	echo -e $text >> $result_file
done

echo "all done"

