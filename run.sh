#!/bin/bash
#SBATCH --job-name=run_rl
#SBATCH --output=run.out
#SBATCH --mail-user=simone.bordoni@uniroma1.it
#SBATCH --mail-type=ALL

python ./simulation/rb.py