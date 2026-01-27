#!/bin/bash

mkdir -p logs

# === Batch 3 ===
# Environment: N3il_with_symmetry
# Symmetric action: default (passing 'rotation_180' as placeholder)
# i = 150, 200, ..., 1000 (step 50)
echo "=== Starting Batch 3 ==="
for i in {150..1000..50}; do
  start=$i
  end=$((i+1))
  step=1
  repeat=1
  # Environment N3il_with_symmetry ignores symmetric_action, but the script expects an argument
  sym_action="rotation_180"
  env="N3il_with_symmetry"
  
  echo "Submitting Batch 3: start=${start}, env=${env}"
  sbatch -c 1 --mem=6G test_mcts_largest_complete_set.sub "${start}" "${end}" "${step}" "${repeat}" "${sym_action}" "${env}"
done


# === Batch 4 ===
# Environment: N3il_with_symmetry_and_symmetric_actions
# Symmetric actions: list
# i = 150, 200, ..., 1000 (step 50)
echo "=== Starting Batch 4 ==="
actions=('rotation_180' 'vertical_flip' 'diagonal_flip' 'rotation_90_then_rotation_180' 'vertical_flip_then_horizontal_flip' 'diagonal_flip_then_anti_diagonal_flip' 'vertical_flip_then_horizontal_flip_then_diagonal_flip')
env="N3il_with_symmetry_and_symmetric_actions"

for sa in "${actions[@]}"; do
  for i in {150..1000..50}; do
    start=$i
    end=$((i+1))
    step=1
    repeat=1
    
    echo "Submitting Batch 4: start=${start}, action=${sa}, env=${env}"
    sbatch -c 1 --mem=6G test_mcts_largest_complete_set.sub "${start}" "${end}" "${step}" "${repeat}" "${sa}" "${env}"
  done
done
