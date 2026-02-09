#!/bin/bash

# Resubmit failed jobs: M4_n30_t1, M4_n40_t1
# Reason: OUT_OF_MEMORY (Increasing memory to 12G)

echo "Resubmitting M4_n30_t1 with 12G..."
sbatch --mem=12G test_mcts_largest_complete_set.sub \
    "30" "1" "M4" \
    "N3il_with_symmetry_and_symmetric_actions" \
    "rotation_90_then_rotation_180" \
    "None" \
    "False" \
    "M4: M3 + Batch Action (C4)"

echo "Resubmitting M4_n40_t1 with 12G..."
sbatch --mem=12G test_mcts_largest_complete_set.sub \
    "40" "1" "M4" \
    "N3il_with_symmetry_and_symmetric_actions" \
    "rotation_90_then_rotation_180" \
    "None" \
    "False" \
    "M4: M3 + Batch Action (C4)"

echo "Done."
