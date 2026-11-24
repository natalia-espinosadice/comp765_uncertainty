#!/bin/bash
# Run once on a Mila LOGIN node:  bash slurm/setup_mila.sh
# Uses the CUDA 12.6 PyTorch build: the default (CUDA 13) wheel has no kernels for Mila's V100s.
set -euo pipefail
module load python/3.10
VENV=$HOME/venvs/uwm-cu126
python -m venv $VENV
source $VENV/bin/activate
pip install --upgrade pip
pip install torch==2.14.1 --index-url https://download.pytorch.org/whl/cu126
pip install "minari[hf,hdf5]" "gymnasium[mujoco]" numpy scipy scikit-learn pandas matplotlib
pip install -e "$(dirname "$0")/.."
export MINARI_DATASETS_PATH=$SCRATCH/minari
# pre-download datasets so compute jobs never depend on the network
python - <<'PY'
import minari
for env in ["halfcheetah", "hopper", "walker2d"]:
    for q in ["simple", "medium", "expert"]:
        ds = minari.load_dataset(f"mujoco/{env}/{q}-v0", download=True)
        print(env, q, ds.total_episodes, ds.total_steps)
PY
python slurm/make_grid.py
python -c "import torch; print('torch', torch.__version__, 'cuda', torch.version.cuda)"
echo "SETUP DONE"
