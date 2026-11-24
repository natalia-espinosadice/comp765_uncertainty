#!/bin/bash
# Run once on a Mila LOGIN node:  bash slurm/setup_mila.sh
set -euo pipefail
module load python/3.10
python -m venv $HOME/venvs/uwm
source $HOME/venvs/uwm/bin/activate
pip install --upgrade pip
pip install torch "minari[hf,hdf5]" "gymnasium[mujoco]" numpy scipy scikit-learn pandas matplotlib
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
