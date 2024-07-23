#!/bin/bash
#SBATCH --job-name=run_rl
#SBATCH --output=run.out
#SBATCH --partition qw11q

python ./simulation/rb_dataset_generator.py