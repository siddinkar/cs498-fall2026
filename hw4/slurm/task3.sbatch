#!/bin/bash
#SBATCH --job-name=hw4_task3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate cs498-hw4
cd /scratch/sdinkar2/cs498-fall2026/hw4

python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -u task3_train_mesh.py "$@"
