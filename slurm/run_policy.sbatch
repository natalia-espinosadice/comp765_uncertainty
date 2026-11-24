#!/bin/bash
# Submit with:  sbatch --array=0-125%20 slurm/run_policy.sbatch
#SBATCH --job-name=uwm-pol
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=8:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
module load python/3.10
source ${UWM_VENV:-$HOME/venvs/uwm-cu126}/bin/activate
export MINARI_DATASETS_PATH=$SCRATCH/minari
export MUJOCO_GL=egl
cd "$SLURM_SUBMIT_DIR"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/policy_grid.txt)
IFS='|' read -r NAME WM EXTRA <<< "$LINE"
OUT=$SCRATCH/uwm_policy/$NAME
echo "[$NAME] wm=$WM $EXTRA"
if [ ! -f "$OUT/result.json" ]; then
  python -m uwm.policy --wm "$SCRATCH/uwm_runs/$WM" --out "$OUT" $EXTRA
fi
