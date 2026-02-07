#!/bin/bash

# Resubmit failed job: M4_n30_t1
# Reason: OUT_OF_MEMORY (Ignored for now)
echo "Resubmitting M4_n30_t1..."

# Config for M4:
# n=30, trial=1
# Env: N3il_with_symmetry_and_symmetric_actions
# Sym Action: rotation_90_then_rotation_180
# Max Sym Level: None
# Save Opt: False
# Comment: M4: M3 + Batch Action (C4)
# Algo: Default (MCTS)

sbatch test_mcts_largest_complete_set.sub \
    "30" "1" "M4" \
    "N3il_with_symmetry_and_symmetric_actions" \
    "rotation_90_then_rotation_180" \
    "None" \
    "False" \
    "M4: M3 + Batch Action (C4)"

echo "Done."
