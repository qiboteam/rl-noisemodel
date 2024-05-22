#!/bin/bash
#SBATCH --job-name=rl-train_low
#SBATCH --output=train_low.out
#SBATCH --mail-user=simone.bordoni@uniroma1.it
#SBATCH --mail-type=ALL

python ./simulation/model_train.py