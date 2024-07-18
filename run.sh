#!/bin/bash
#SBATCH --job-name=run_rl
#SBATCH --output=run.out
#SBATCH --partition iqm5q

python ./simulation/rb.py