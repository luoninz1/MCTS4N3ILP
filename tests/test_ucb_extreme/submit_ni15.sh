#!/bin/bash

# Configuration for n=15 (ni15_Env)
start=15
end=104
step=20
repeat=10
sym_action="None"
env="No_isosceles"

echo "Submitting missing job: start=${start}, env=${env}"
sbatch -c 1 --mem=6G test_no_isosceles.sub "${start}" "${end}" "${step}" "${repeat}" "${sym_action}" "${env}"
