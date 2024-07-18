#!/bin/bash
#SBATCH --job-name=rl-dataset_size
#SBATCH --output=run.out
#SBATCH --partition iqm5q
#SBATCH --mail-user=simone.bordoni@uniroma1.it
#SBATCH --mail-type=ALL

python ./simulation/rb_dataset_generator.py