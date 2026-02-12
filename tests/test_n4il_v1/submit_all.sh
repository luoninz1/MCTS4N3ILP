#!/bin/bash

mkdir -p logs

# === Test N4il ===
# Environment: N4il
# Symmetric action: rotation_90_then_rotation_180
# i = 3 ~ 22
echo "=== Starting N4il Tests ==="

start=3
end=23  # end value for loop is exclusive in loop declaration, but here we iterate i
# Actually user said "n_start=3~22 提交 20个job", so n=3, 4, ..., 22

for i in {3..22}; do
  start=$i
  end=104
  step=20
  repeat=10
  
  sym_action="rotation_90_then_rotation_180"
  env="N4il"
  
  echo "Submitting N4il Job: start=${start}, end=${end}, action=${sym_action}"
  
  # Submit job
  sbatch test_n4il.sub "${start}" "${end}" "${step}" "${repeat}" "${sym_action}" "${env}"
  
  sleep 0.5
done
