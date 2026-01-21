#!/bin/bash
# mem(start) = 6G + floor((start-53)/20)*6G
# cpus(start) = 1 + floor((start-53)/20)

for start in {53..100}; do
  inc=$(( (start-53)/20 ))
  if (( inc < 0 )); then inc=0; fi

  mem=$(( 6 + inc*6 ))     # GiB
  cpus=$(( 1 + inc ))      # cores

  echo "Submitting start=${start}  ->  CPUs=${cpus}, MEM=${mem}G"
  sbatch -c "${cpus}" --mem="${mem}G" test_mcts_largest_complete_set.sub "${start}"
done