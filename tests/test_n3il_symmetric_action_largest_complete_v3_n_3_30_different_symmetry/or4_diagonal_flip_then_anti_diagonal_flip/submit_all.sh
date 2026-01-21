#!/bin/bash

for start in {3..7}; do
  echo "Submitting start=${start}  ->  CPUs=1, MEM=6G"
  sbatch -c 1 --mem=6G test_mcts_largest_complete_set.sub "${start}"
done