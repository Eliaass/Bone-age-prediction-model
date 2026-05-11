#!/bin/bash

#SBATCH --mail-user=elias.benchouk@students.unibe.ch
#SBATCH --mail-type=BEGIN,FAIL,END

#SBATCH --job-name=train_efficientnetv2_m
#SBATCH --output=/storage/homefs/eb19p026/DL_Project/logs/%x_%j.out
#SBATCH --error=/storage/homefs/eb19p026/DL_Project/logs/%x_%j.err

#SBATCH --time=8:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=8
#SBATCH --mem=64G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:a100:1

set -euo pipefail

echo "=== Job started ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Date: $(date)"S

module purge
module load Python/3.11.3-GCCcore-12.3.0
module load CUDA/12.6.0

echo "=== Loaded modules ==="
module list

PROJECT_DIR=/storage/homefs/eb19p026/DL_Project
cd "$PROJECT_DIR"

mkdir -p logs

if [ ! -d "$PROJECT_DIR/venv" ]; then
    echo "Creating virtual environment..."
    python3 -m venv "$PROJECT_DIR/venv"
fi

source "$PROJECT_DIR/venv/bin/activate"

echo "=== Python environment ==="
echo "CWD: $(pwd)"
echo "Python: $(which python)"
python --version

python -m pip install --upgrade pip
python -m pip install -r requirements.txt

echo "=== CUDA / PyTorch check ==="
python - <<'PY'
import torch
print("torch:", torch.__version__)
print("torch.version.cuda:", torch.version.cuda)
print("cuda available:", torch.cuda.is_available())
if torch.cuda.is_available():
    print("gpu:", torch.cuda.get_device_name(0))
    print("gpu memory GB:", round(torch.cuda.get_device_properties(0).total_memory / 1024**3, 2))
PY

echo "=== nvidia-smi ==="
nvidia-smi

echo "=== Starting EfficientNetV2-M final training ==="

python efficientnetv2-m_training.py

echo "=== EfficientNetV2-M final training finished ==="
echo "Date: $(date)"