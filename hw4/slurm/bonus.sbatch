#!/bin/bash
#SBATCH --job-name=hw4_bonus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=16G
#SBATCH --time=00:30:00
#SBATCH --output=slurm/%x_%j.out

set -e
export OMP_NUM_THREADS=$SLURM_CPUS_PER_TASK

eval "$(conda shell.bash hook)"
conda activate cs498-hw4
cd /scratch/sdinkar2/cs498-fall2026/hw4

python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -m unittest tests.test_bonus_ball_insertion
python -u bonus_ball_insertion.py --mode check
python -u bonus_ball_insertion.py --mode render --checkpoint outputs/task3/lego_train/model.pt "$@"
