#!/bin/bash
#SBATCH --job-name=run_rl
#SBATCH --output=rb_dataset.out
#SBATCH --partition qw11q

#python circuit_test.py
python ./simulation/rb_dataset_generator.py