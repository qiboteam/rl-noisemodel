#!/bin/bash
#SBATCH --job-name=run_rl
#SBATCH --output=rb.out
#SBATCH --partition qw11q

#python circuit_test.py
python ./simulation/rb.py