"""Diagnostic: are the policies bad, or only bad from the env's reset state?

Imagined rollouts start 31+ steps into a dataset episode (after a 32-step true history), so a
policy never sees the env's initial standing pose during training. This script evaluates each
saved policy in the real env twice:
  reset  from env.reset(), as in policy.py
  data   from states taken mid-episode from the training split (env state set from the observation;
         the x position, which the observation omits, is set to 0)
and reports reward per step for both. If 'data' >> 'reset', the failure is the start-state
mismatch, not model exploitation.

python -m uwm.eval_from_data --runs $SCRATCH/uwm_policy --wm-root $SCRATCH/uwm_runs --out start_diag.csv
"""
from __future__ import annotations

import argparse
import json
import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .data import Normalizer, load_source, split_episodes
from .policy import OBS_CLIP, ActorCritic, env_kind, make_real_env

NQ_SKIP = {"halfcheetah": 8, "hopper": 5, "walker2d": 8}   # obs = qpos[1:] (this many) + qvel


def set_state_from_obs(env, kind, obs):
    u = env.unwrapped
    n = NQ_SKIP[kind]
    qpos = np.concatenate([[0.0], obs[:n]])
    qvel = obs[n:n + u.model.nv]
    u.set_state(qpos, qvel)
    return u._get_obs()


@torch.no_grad()
def run(ac, env, kind, norm, starts, steps, seed):
    per_step, lens = [], []
    for i, s in enumerate(starts):
        o, _ = env.reset(seed=seed + i)
        if s is not None:
            o = set_state_from_obs(env, kind, s)
        R, T = 0.0, 0
        for _ in range(steps):
            x = torch.as_tensor(np.clip(norm.obs(o), -OBS_CLIP, OBS_CLIP), dtype=torch.float32)
            o, r, term, trunc, _ = env.step(ac.pi(x).clamp(-1, 1).numpy())
            R += float(r); T += 1
            if term:
                break
        per_step.append(R / T); lens.append(T)
    return float(np.mean(per_step)), float(np.mean(lens))


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--wm-root", required=True)
    p.add_argument("--episodes", type=int, default=5)
    p.add_argument("--steps", type=int, default=500)
    p.add_argument("--filter", default="", help="regex on run name")
    p.add_argument("--out", default="start_diag.csv")
    a = p.parse_args(argv)

    cache, rows = {}, []
    for f in sorted(Path(a.runs).glob("*/result.json")):
        if not re.search(a.filter, f.parent.name):
            continue
        r = json.loads(f.read_text())
        wm = Path(r["args"]["wm"]).name
        wa = r["wm_args"]; kind = env_kind(wa["train"])
        if wm not in cache:
            with open(Path(a.wm_root) / wm / "norm.pkl", "rb") as fh:
                norm = Normalizer.from_state(pickle.load(fh))
            eps = load_source(wa["train"], None, seed=wa["seed"])
            tr, _ = split_episodes(eps, wa["test_frac"], seed=12345)
            rng = np.random.default_rng(0)
            pool = [e["obs"][rng.integers(50, len(e["obs"]) - 1)] for e in tr if len(e["obs"]) > 60]
            starts = [pool[i] for i in rng.choice(len(pool), a.episodes, replace=False)]
            data_ps = float(np.mean([e["rew"].mean() for e in tr]))
            cache[wm] = (norm, starts, data_ps, make_real_env(wa["train"]))
        norm, starts, data_ps, env = cache[wm]
        ck = torch.load(f.parent / "policy.pt", map_location="cpu", weights_only=False)
        ac = ActorCritic(ck["D"], ck["A"]); ac.load_state_dict(ck["state"]); ac.eval()
        ps_reset, len_reset = run(ac, env, kind, norm, [None] * a.episodes, a.steps, 0)
        ps_data, len_data = run(ac, env, kind, norm, starts, a.steps, 100)
        row = dict(run=f.parent.name, env=kind, dataset_per_step=data_ps,
                   reset_per_step=ps_reset, reset_len=len_reset, data_per_step=ps_data, data_len=len_data)
        rows.append(row); print(json.dumps(row), flush=True)
    df = pd.DataFrame(rows); df.to_csv(a.out, index=False)
    with pd.option_context("display.width", 200):
        print(df.groupby("env")[["dataset_per_step", "reset_per_step", "data_per_step", "reset_len", "data_len"]]
              .mean().round(2))


if __name__ == "__main__":
    main()
