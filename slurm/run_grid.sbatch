#!/bin/bash
# Submit with:  sbatch --array=0-$(( $(wc -l < slurm/grid.txt) - 1 ))%20 slurm/run_grid.sbatch
#SBATCH --job-name=uwm
#SBATCH --partition=long
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=4
#SBATCH --mem=24G
#SBATCH --time=4:00:00
#SBATCH --output=logs/%x_%A_%a.out
set -euo pipefail
module load python/3.10
source $HOME/venvs/uwm/bin/activate
export MINARI_DATASETS_PATH=$SCRATCH/minari
cd "$SLURM_SUBMIT_DIR"

LINE=$(sed -n "$((SLURM_ARRAY_TASK_ID + 1))p" slurm/grid.txt)
IFS='|' read -r NAME TRAIN OOD EXTRA <<< "$LINE"
OUT=$SCRATCH/uwm_runs/$NAME
echo "[$NAME] $TRAIN  $EXTRA"

if [ ! -f "$OUT/model.pt" ]; then
  python -m uwm.train --train "$TRAIN" --out "$OUT" $EXTRA
fi
# evaluate every model under both rollout-feedback modes (cheap, isolates train vs test effect)
for FB in mean member; do
  python -m uwm.evaluate --run "$OUT" --ood $OOD --feedback $FB
done
