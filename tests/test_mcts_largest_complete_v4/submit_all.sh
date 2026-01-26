#!/bin/bash
# mem(start) = 6G + floor((start-53)/20)*6G
# cpus(start) = 1 + floor((start-53)/20)

for start in {73..103}; do
  echo "Submitting start=${start}  ->  CPUs=1, MEM=6G"
  sbatch -c 1 --mem=6G test_mcts_largest_complete_set.sub "${start}"
done