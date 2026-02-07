#!/bin/bash

mkdir -p logs

echo "Starting Ablation Study Submissions (Tree Reuse Variant)..."

# Loop over grid sizes
for n in 30 40 50; do
    echo "=== Grid Size n=$n ==="
    
    # Loop over trials (0 to 9)
    for trial in {0..9}; do
        echo "  Submitting Trial $trial..."
        
        # O-tr: Ours + Tree Reuse
        # Env: N3il_with_symmetry_and_symmetric_actions
        # Sym Action: rotation_90_then_rotation_180
        # Max Sym: None (Full)
        # Save Opt: True
        # Algo: MCTS_Tree_Reuse
        
        sbatch test_mcts_largest_complete_set.sub \
            "$n" "$trial" "O-tr" \
            "N3il_with_symmetry_and_symmetric_actions" \
            "rotation_90_then_rotation_180" \
            "None" \
            "True" \
            "O-tr: Ours (Full) + Tree Reuse" \
            "MCTS_Tree_Reuse"
            
        sleep 0.5
        
    done
done
