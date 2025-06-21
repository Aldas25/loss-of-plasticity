#!/bin/bash

pairs="2,1 21,2"
pairs="$pairs 23,0 23,1 23,2 23,3 23,4"
pairs="$pairs 24,0 24,1 24,2 24,3 24,4"
pairs="$pairs 25,0 25,1 25,2 25,3 25,4"
pairs="$pairs 26,0 26,1 26,2 26,3 26,4"

for pair in $pairs; do
	IFS=',' read -r first second <<< "$pair"
	index_file=$(($first * 5 + $second))
	
	#echo "pair: $first $second,  index file: $index_file"
	sbatch ./slurm_jobs/cbp_snp_$index_file.slurm
done

