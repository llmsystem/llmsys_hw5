#!/usr/bin/env bash
#SBATCH --job-name=hw56-grade
#SBATCH --partition=GPU-shared
#SBATCH --gres=gpu:v100-32:2
#SBATCH --cpus-per-task=5
#SBATCH --mem=32G
#SBATCH --time=00:20:00
#SBATCH --output=hw56-grade-%j.log
set -euo pipefail
# Submit from the repository: sbatch -A YOUR_GPU_ALLOCATION scripts/grade_bridges.sbatch
cd "${SLURM_SUBMIT_DIR}"
CORE_PYTHON="${CORE_PYTHON:-${PWD}/.venv/bin/python}"
"${CORE_PYTHON}" grade.py --device cuda --output "artifacts/grade-${SLURM_JOB_ID}.json"
