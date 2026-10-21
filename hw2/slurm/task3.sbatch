#!/bin/bash
#SBATCH --job-name=hw2_task3
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=01:00:00
#SBATCH --output=slurm/%x_%j.out

module load python/3.11.11
source /scratch/sdinkar2/cs498-fall2026/.venv/bin/activate
unset PYTHONPATH
cd /scratch/sdinkar2/cs498-fall2026/hw2

python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python task3_transformer.py
