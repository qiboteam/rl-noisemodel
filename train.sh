#!/bin/bash
#SBATCH --job-name=rl-train
#SBATCH --output=train.out
#SBATCH --mail-user=simone.bordoni@uniroma1.it
#SBATCH --mail-type=END  # Send email when job ends

python ./simulation/model_train.py