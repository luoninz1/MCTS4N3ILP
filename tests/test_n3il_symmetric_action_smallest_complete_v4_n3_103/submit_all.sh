#!/bin/bash

mkdir -p logs

# === Batch 1 ===
# Environment: N3il_with_symmetry
# Symmetric action: None (str will be converted to None in python code)
# i = 3 ~ 12
echo "=== Starting Batch 1 ==="
for i in {3..12}; do
  start=$i
  end=104
  step=10
  repeat=10
  # Using default symmetric action value, but environment is N3il_with_symmetry
  # symmetric_action arg is still required by our .sub script, so we pass the default 'rotation_180'
  sym_action="None"
  env="N3il_with_symmetry"
  
  echo "Submitting Batch 1: start=${start}, env=${env}"
  sbatch -c 1 --mem=6G test_mcts_smallest_complete_set.sub "${start}" "${end}" "${step}" "${repeat}" "${sym_action}" "${env}"
  sleep 0.5
done


# === Batch 2 ===
# Environment: N3il_with_symmetry_and_symmetric_actions
# Symmetric actions: list
# i = 3 ~ 12
echo "=== Starting Batch 2 ==="
actions=('rotation_180' 'vertical_flip' 'diagonal_flip' 'rotation_90_then_rotation_180' 'vertical_flip_then_horizontal_flip' 'diagonal_flip_then_anti_diagonal_flip' 'vertical_flip_then_horizontal_flip_then_diagonal_flip')
env="N3il_with_symmetry_and_symmetric_actions"

for sa in "${actions[@]}"; do
  for i in {3..12}; do
    start=$i
    end=104
    step=10
    repeat=10
    
    echo "Submitting Batch 2: start=${start}, action=${sa}, env=${env}"
    sbatch -c 1 --mem=6G test_mcts_smallest_complete_set.sub "${start}" "${end}" "${step}" "${repeat}" "${sa}" "${env}"
    sleep 0.5
  done
done