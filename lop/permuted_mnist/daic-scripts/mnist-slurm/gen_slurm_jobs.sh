#!/bin/bash

output_dir=slurm_jobs

mkdir -p $output_dir

algo=$1
index_from=$2
index_to=$3

echo "doing for $algo indices: [$index_from, $index_to]"

	
for ((i=$index_from; i<=$index_to; i++)); do
	task="mnist_$algo"
	run_id="$task-$i"
	conf_folder_suf="_temp_cfg"
	conf_folder="$algo$conf_folder_suf"
	result_file=$output_dir/$run_id.slurm
	cp mnist_sbatch-template.slurm $result_file
	text="#SBATCH --output=/home/nfs/alenksas/slurm_outputs/$run_id.%j.out"
	text="$text\n#SBATCH --error=/home/nfs/alenksas/slurm_outputs/$run_id.%j.err"
	text="$text\nmodule use /opt/insy/modulefiles"
	text="$text\nmodule load miniconda"
	text="$text\nconda activate lop"
	text="$text\nsrun ./mnist_one_run.sh $conf_folder $i $run_id"
	echo -e $text >> $result_file
done

echo "all done"

