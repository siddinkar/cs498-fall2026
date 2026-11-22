#!/bin/bash
#SBATCH --job-name=hw3_train
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=06:00:00
#SBATCH --output=slurm/%x_%j.out

module load python/3.11.11
source /scratch/sdinkar2/cs498-fall2026/.venv/bin/activate
unset PYTHONPATH
cd /scratch/sdinkar2/cs498-fall2026/hw3

python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -u task3_detector.py --workers 4 "$@"
python -u task4_inference.py
