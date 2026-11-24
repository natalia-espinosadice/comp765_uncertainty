"""Quick GPU sanity check: one forward+backward of the world model on the current GPU.

srun --partition=main --gres=gpu:1 --constraint=volta -c 2 --mem=8G -t 0:10:00 \
    ~/venvs/uwm-cu126/bin/python slurm/gpu_check.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))
import torch  # noqa: E402

from uwm.model import EnsembleWorldModel  # noqa: E402

dev = torch.device("cuda")
print("torch", torch.__version__, "| cuda", torch.version.cuda, "|", torch.cuda.get_device_name(),
      "SM", torch.cuda.get_device_capability(), "| cudnn on:", torch.backends.cudnn.enabled)
for trunk in ("shared", "independent"):
    m = EnsembleWorldModel(17, 6, 5, trunk).to(dev)
    mu, ls, r = m.rollout(torch.randn(64, 32, 17, device=dev), torch.randn(64, 40, 6, device=dev), 8)
    (mu.square().mean() + ls.mean() + r.mean()).backward()
    torch.cuda.synchronize()
print("GPU CHECK OK")
