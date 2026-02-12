#!/bin/bash

mkdir -p logs

# === Batch 1 ===
# Environment: GeometricDominating
# Symmetric actions: list
# i = 3 ~ 12
echo "=== Starting Batch 1 ==="
actions=('None' 'rotation_180' 'vertical_flip' 'diagonal_flip' 'rotation_90_then_rotation_180' 'vertical_flip_then_horizontal_flip' 'diagonal_flip_then_anti_diagonal_flip' 'vertical_flip_then_horizontal_flip_then_diagonal_flip')
env="GeometricDominating"

for sa in "${actions[@]}"; do
  for i in {3..12}; do
    start=$i
    end=104
    step=10
    repeat=10
    
    echo "Submitting Batch 1: start=${start}, action=${sa}, env=${env}"
    sbatch -c 1 --mem=6G test_geodom.sub "${start}" "${end}" "${step}" "${repeat}" "${sa}" "${env}"
    sleep 0.5
  done
done