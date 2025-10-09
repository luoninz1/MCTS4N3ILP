#!/bin/bash
for start in {50..103}; do
    sbatch test_mcts_smallest_complete_set.sub "$start" 50
done