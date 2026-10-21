#!/bin/bash
#SBATCH --job-name=hw2_bonus
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=slurm/%x_%j.out

module load python/3.11.11
source /scratch/sdinkar2/cs498-fall2026/.venv-utonia/bin/activate
unset PYTHONPATH
cd /scratch/sdinkar2/cs498-fall2026/hw2
export HF_HUB_OFFLINE=1

python -c "import torch; print(torch.__version__, 'CUDA:', torch.cuda.is_available())"
python -u bonus_utonia.py --checkpoint ~/.cache/utonia/ckpt/utonia.pth --in-memory "$@"
