#!/bin/bash
#SBATCH --job-name=hw3_test
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

module load python/3.11.11
source /scratch/sdinkar2/cs498-fall2026/.venv/bin/activate
unset PYTHONPATH
cd /scratch/sdinkar2/cs498-fall2026/hw3

python -u task4_inference.py --test
