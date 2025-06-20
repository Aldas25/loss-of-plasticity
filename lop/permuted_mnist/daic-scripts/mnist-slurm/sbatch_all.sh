#!/bin/bash

output_dir=slurm_jobs

for res_file in $output_dir/*; do
#for ((i=0; i<30; i++)); do
	#res_file=$output_dir/job_$i.slurm
	echo "going to sbatch file $res_file"
	sbatch $res_file
done
