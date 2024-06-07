#!/bin/bash
#SBATCH --job-name=rl-train_1q_5
#SBATCH --output=train_1q_5.out
#SBATCH --mail-user=simone.bordoni@uniroma1.it
#SBATCH --mail-type=ALL

python ./simulation/model_train.py