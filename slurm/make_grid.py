"""Write slurm/grid.txt: one job per line as 'NAME|TRAIN_SPEC|OOD_SPECS|EXTRA_ARGS'."""
from pathlib import Path

ENVS = ["halfcheetah", "hopper", "walker2d"]
SEEDS = [0, 1, 2]
rows = []

def add(env, name, extra):
    tr = f"minari:mujoco/{env}/medium-v0"
    ood = f"minari:mujoco/{env}/simple-v0 minari:mujoco/{env}/expert-v0"
    for s in SEEDS:
        rows.append(f"{env}_{name}_s{s}|{tr}|{ood}|{extra} --seed {s}")

# Core 2x2 ablation on all envs: trunk sharing x data bootstrap
for env in ENVS:
    for trunk in ["shared", "independent"]:
        for boot in ["none", "poisson"]:
            add(env, f"{trunk}_{boot}", f"--trunk {trunk} --bootstrap {boot}")

# Extra ablations on HalfCheetah only
env = "halfcheetah"
for m in [2, 10]:
    for trunk in ["shared", "independent"]:
        add(env, f"{trunk}_none_E{m}", f"--trunk {trunk} --bootstrap none --members {m}")
for trunk in ["shared", "independent"]:
    add(env, f"{trunk}_none_fbmember", f"--trunk {trunk} --bootstrap none --feedback member")
add(env, "shared_none_mse", "--trunk shared --bootstrap none --loss mse")

Path(__file__).with_name("grid.txt").write_text("\n".join(rows) + "\n")
print(len(rows), "jobs")
