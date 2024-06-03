#!/bin/bash
#SBATCH --job-name=rl-dataset_size
#SBATCH --output=dataset_size.out
#SBATCH --mail-user=simone.bordoni@uniroma1.it
#SBATCH --mail-type=ALL

python ./simulation/test_dataset_size.py