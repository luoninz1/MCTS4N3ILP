#!/bin/bash

mkdir -p logs

# === Batch 1 ===
# Environment: No_isosceles
# Symmetric action: None (str will be converted to None in python code)
# i = 3 ~ 22
echo "=== Starting Batch 1 ==="
for i in {3..22}; do
  start=$i
  end=104
  step=20
  repeat=10
  # Using default symmetric action value, but environment is No_isosceles
  # symmetric_action arg is still required by our .sub script, so we pass the default 'rotation_180'
  sym_action="None"
  env="No_isosceles"
  
  echo "Submitting Batch 1: start=${start}, env=${env}"
  sbatch -c 1 --mem=6G test_no_isosceles.sub "${start}" "${end}" "${step}" "${repeat}" "${sym_action}" "${env}"
  sleep 0.5
done