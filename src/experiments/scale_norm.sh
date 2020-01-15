#!/bin/bash

#SBATCH --ntasks=1
#SBATCH --cpus-per-task=16
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2019_hsic_align/
#SBATCH --job-name=hsic_scale
#SBATCH --output=/home/emmanuel/projects/2019_hsic_align/src/experiments/logs/scale-job-%j.log

module load Anaconda3
source activate it4dnn

# Global Experiment

# Individual
srun --nodes 1 --ntasks 1 python -u src/experiments/scale_normalize.py --case 1 --save v2 &
srun --nodes 1 --ntasks 1 python -u src/experiments/scale_normalize.py --case 2 --save v2 &
srun --nodes 1 --ntasks 1 python -u src/experiments/scale_normalize.py --case 3 --save v2 &
srun --nodes 1 --ntasks 1 python -u src/experiments/scale_normalize.py --case 4 --save v2
