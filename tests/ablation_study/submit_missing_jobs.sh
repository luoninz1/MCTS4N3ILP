#!/bin/bash

# Resubmit missing jobs for ablation study
# Memory set to 18G as requested

echo "Starting resubmission of missing jobs with 6G memory..."

# ==========================================
# M4: M3 + Batch Action (C4)
# Env: N3il_with_symmetry_and_symmetric_actions
# Sym Action: rotation_90_then_rotation_180
# Max Sym: None
# Save Opt: False
# ==========================================

# n=40, seeds 1, 6
for seed in 1 6; do
    echo "Submitting M4 n=40 seed=$seed"
    sbatch --mem=6G test_mcts_largest_complete_set.sub \
        "40" "$seed" "M4" \
        "N3il_with_symmetry_and_symmetric_actions" \
        "rotation_90_then_rotation_180" \
        "None" \
        "False" \
        "M4: M3 + Batch Action (C4)"
    sleep 0.5
done

# n=50, seeds 0, 1
for seed in 0 1; do
    echo "Submitting M4 n=50 seed=$seed"
    sbatch --mem=6G test_mcts_largest_complete_set.sub \
        "50" "$seed" "M4" \
        "N3il_with_symmetry_and_symmetric_actions" \
        "rotation_90_then_rotation_180" \
        "None" \
        "False" \
        "M4: M3 + Batch Action (C4)"
    sleep 0.5
done

# ==========================================
# Ours (Full)
# Env: N3il_with_symmetry_and_symmetric_actions
# Sym Action: rotation_90_then_rotation_180
# Max Sym: None
# Save Opt: True
# ==========================================

# n=40, seeds 3, 4, 6
for seed in 3 4 6; do
    echo "Submitting Ours n=40 seed=$seed"
    sbatch --mem=6G test_mcts_largest_complete_set.sub \
        "40" "$seed" "Ours" \
        "N3il_with_symmetry_and_symmetric_actions" \
        "rotation_90_then_rotation_180" \
        "None" \
        "True" \
        "Ours (Full)"
    sleep 0.5
done

# n=50, seed 1
echo "Submitting Ours n=50 seed=1"
sbatch --mem=6G test_mcts_largest_complete_set.sub \
    "50" "1" "Ours" \
    "N3il_with_symmetry_and_symmetric_actions" \
    "rotation_90_then_rotation_180" \
    "None" \
    "True" \
    "Ours (Full)"

echo "All missing jobs submitted."
