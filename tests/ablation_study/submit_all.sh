#!/bin/bash

mkdir -p logs

echo "Starting Ablation Study Submissions..."

# Loop over grid sizes
for n in 30 40 50; do
    echo "=== Grid Size n=$n ==="
    
    # Loop over trials (0 to 9)
    for trial in {0..9}; do
        echo "  Submitting Trial $trial..."
        
        # M0: Baseline/Vanilla (FVAS)
        # Env: N3il_with_FVAS
        sbatch test_mcts_largest_complete_set.sub \
            "$n" "$trial" "M0" \
            "N3il_with_FVAS" "None" "None" "False" \
            "M0: Baseline/Vanilla (FVAS)"
        sleep 0.5

        # M1: M0 + SVAS
        # Env: N3il_with_SVAS_wo_inc
        sbatch test_mcts_largest_complete_set.sub \
            "$n" "$trial" "M1" \
            "N3il_with_SVAS_wo_inc" "None" "None" "False" \
            "M1: M0 + SVAS"
        sleep 0.5
        
        # M2: M1 + Incremental AS Update
        # Env: N3il (Standard env typically has incremental updates)
        sbatch test_mcts_largest_complete_set.sub \
            "$n" "$trial" "M2" \
            "N3il" "None" "None" "False" \
            "M2: M1 + Incremental AS Update"
        sleep 0.5
        
        # M3: M2 + Dihedral Pruning
        # Env: N3il_with_symmetry, max_symmetry_level=1
        sbatch test_mcts_largest_complete_set.sub \
            "$n" "$trial" "M3" \
            "N3il_with_symmetry" "None" "1" "False" \
            "M3: M2 + Dihedral Pruning"
        sleep 0.5
        
        # M4: M3 + Batch Action (C4)
        # Env: N3il_with_symmetry_and_symmetric_actions, symmetric_action=rotation_90_then_rotation_180, max_sym=None
        sbatch test_mcts_largest_complete_set.sub \
            "$n" "$trial" "M4" \
            "N3il_with_symmetry_and_symmetric_actions" "rotation_90_then_rotation_180" "None" "False" \
            "M4: M3 + Batch Action (C4)"
        sleep 0.5
        
        # Ours (Full): M4 + Optimal Record
        # Same as M4, but save_optimal_terminal_state=True
        sbatch test_mcts_largest_complete_set.sub \
            "$n" "$trial" "Ours" \
            "N3il_with_symmetry_and_symmetric_actions" "rotation_90_then_rotation_180" "None" "True" \
            "Ours (Full)"
        sleep 0.5
        
    done
done
