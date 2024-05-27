#!/bin/bash
#SBATCH --job-name=rl-train_2
#SBATCH --output=train_2.out
#SBATCH --mail-user=simone.bordoni@uniroma1.it
#SBATCH --mail-type=ALL

python ./simulation/model_train.py