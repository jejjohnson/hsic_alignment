#!/bin/bash
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=28
#SBATCH --exclude=nodo17
#SBATCH --workdir=/home/emmanuel/projects/2019_hsic_align/
#SBATCH --job-name=hsic_dist
#SBATCH --output=/home/emmanuel/projects/2019_hsic_align/src/experiments/logs/hsic_dist_%j.log

module load Anaconda3
conda activate hsic_align

# Individual
srun --nodes 1 --ntasks 1 python -u src/experiments/distributions.py --dataset gauss --njobs $SLURM_CPUS_PER_TASK --verbose 1 --save v1 &
srun --nodes 1 --ntasks 1 python -u src/experiments/distributions.py --dataset tstudent --njobs $SLURM_CPUS_PER_TASK --verbose 1 --save v1 
