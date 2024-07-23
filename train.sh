#!/bin/bash
#SBATCH --job-name=rl-train
#SBATCH --output=train.out

python ./simulation/model_train.py