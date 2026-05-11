#!/bin/bash

#SBATCH --mail-user=elias.benchouk@students.unibe.ch
#SBATCH --mail-type=BEGIN,FAIL,END

#SBATCH --job-name=hpoV2_efficientnetv2_m_boneage
#SBATCH --output=/storage/homefs/eb19p026/DL_Project/logs/%x_%j.out
#SBATCH --error=/storage/homefs/eb19p026/DL_Project/logs/%x_%j.err

#SBATCH --time=10:00:00
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --cpus-per-task=12
#SBATCH --mem=64G

#SBATCH --partition=gpu
#SBATCH --gres=gpu:h100:1

set -euo pipefail

echo "=== Job started ==="
echo "Job ID: ${SLURM_JOB_ID}"
echo "Host: $(hostname)"
echo "Date: $(date)"

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

echo "=== Starting HPOV2 ==="

python hpo_efficientnetv2_m_boneage.py \
    --project_dir "$PROJECT_DIR" \
    --csv_path "$PROJECT_DIR/boneage-training-dataset.csv" \
    --image_dir "$PROJECT_DIR/cropped_overlayed_RSNA_dataset_1024x1024" \
    --hpo_dir "$PROJECT_DIR/hpo_efficientnetv2_m_boneage_HPOV2" \
    --study_name "hpo_efficientnetv2_m_boneage_HPOV2" \
    --batch_size 32 \
    --grad_accum_steps 1 \
    --num_workers 6 \
    --n_trials 30 \
    --hpo_epochs 20 \
    --hpo_steps_per_epoch 200 \
    --hpo_patience 5 \
    --use_amp

echo "=== HPOV2 finished ==="
echo "Date: $(date)"