"""Write slurm/policy_grid.txt: 'NAME|WM_RUN|EXTRA_ARGS' per line (stretch goal, MOPO-PPO).

3 envs x 2 world models (paper's shared trunk, independent trunks) x 3 seeds x 7 penalty settings.
Policy seed s uses world-model seed s, so model-to-model variance is included in the error bars.
"""
from pathlib import Path

rows = []
PEN = [("none", "--penalty none")]
for pen in ["epi", "ale", "pair"]:
    PEN += [(f"{pen}_norm", f"--penalty {pen} --lam 1"), (f"{pen}_raw", f"--penalty {pen} --lam 1 --raw")]
for env in ["halfcheetah", "hopper", "walker2d"]:
    for wm in ["shared_none", "independent_none"]:
        for s in [0, 1, 2]:
            for tag, extra in PEN:
                rows.append(f"{env}_{wm}_{tag}_s{s}|{env}_{wm}_s{s}|{extra} --seed {s}")
# Penalty-strength sweep (rescaled penalties only): lam=1 turned out far too weak (see README).
lam_rows = []
for env in ["halfcheetah", "hopper", "walker2d"]:
    for wm in ["shared_none", "independent_none"]:
        for s in [0, 1, 2]:
            for pen in ["epi", "ale", "pair"]:
                for lam in [3, 10, 30, 100]:
                    lam_rows.append(f"{env}_{wm}_{pen}_norm_lam{lam}_s{s}|{env}_{wm}_s{s}|--penalty {pen} --lam {lam} --seed {s}")
Path(__file__).with_name("policy_grid_lam.txt").write_text("\n".join(lam_rows) + "\n")
print(len(lam_rows), "lam-sweep jobs")
Path(__file__).with_name("policy_grid.txt").write_text("\n".join(rows) + "\n")
print(len(rows), "jobs")
