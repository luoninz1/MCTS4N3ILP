#!/bin/bash

mkdir -p logs

# === Test No_4_on_circle ===
# Environment: No_4_on_circle
# Symmetric action: None
# i = 3 ~ 22
echo "=== Starting No_4_on_circle Tests ==="

for i in {3..22}; do
  start=$i
  end=104
  step=20
  repeat=10
  
  sym_action="None"
  env="No_4_on_circle"
  
  echo "Submitting No_4_on_circle Job: start=${start}, end=${end}, action=${sym_action}"
  
  # Submit job
  sbatch test_geodom.sub "${start}" "${end}" "${step}" "${repeat}" "${sym_action}" "${env}"
  
  sleep 0.5
done
