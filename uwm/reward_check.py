"""Diagnostic: is the world model wrong about the policy's reward, or is something else broken?

For one saved policy: run it in the REAL env for hist+H steps, then feed the recorded real
observations/actions to the world model and compare its reward predictions with the real rewards:
  teacher-forced  (h=1): model sees the true history at every step, predicts one step ahead
  open-loop       (h=H): model rolls H steps on its own predictions with the policy's real actions
  closed-loop     : the policy acts on the model's own imagined states (what PPO trains on)
Also reported for dataset (behaviour-policy) windows, as a reference for how accurate the reward
head is on in-distribution data.

If teacher-forced predictions on the policy's data are far above the real reward while dataset
predictions are accurate, the policy has found actions where the model is wrong: exploitation.
If the model is also wrong on the dataset, the reward head or the data pipeline is broken.

python -m uwm.reward_check --policy $SCRATCH/uwm_policy/halfcheetah_shared_none_none_s0 \
    --wm-root $SCRATCH/uwm_runs
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch

from .data import Normalizer, WindowSet, load_source, split_episodes
from .model import EnsembleWorldModel
from .policy import OBS_CLIP, ActorCritic, Imagination, make_real_env


@torch.no_grad()
def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--policy", required=True)
    p.add_argument("--wm-root", required=True)
    p.add_argument("--episodes", type=int, default=4)
    p.add_argument("--horizon", type=int, default=50)
    a = p.parse_args(argv)

    r = json.loads((Path(a.policy) / "result.json").read_text())
    wm_dir = Path(a.wm_root) / Path(r["args"]["wm"]).name
    ck = torch.load(wm_dir / "model.pt", map_location="cpu", weights_only=False)
    wa = ck["args"]; M, H = wa["hist"], a.horizon
    model = EnsembleWorldModel(ck["D"], ck["A"], wa["members"], wa["trunk"], wa["feedback"])
    model.load_state_dict(ck["state"]); model.eval()
    with open(wm_dir / "norm.pkl", "rb") as f:
        norm = Normalizer.from_state(pickle.load(f))
    pk = torch.load(Path(a.policy) / "policy.pt", map_location="cpu", weights_only=False)
    ac = ActorCritic(pk["D"], pk["A"]); ac.load_state_dict(pk["state"]); ac.eval()
    env = make_real_env(wa["train"])
    rs = lambda x: x * norm.r_std + norm.r_mean                       # model reward -> real units

    def model_rewards(o_n, act):
        """o_n [B, M+H+1, D] normalised real obs, act [B, M+H, A] -> (tf [B,H], ol [B,H])."""
        tf = []
        for k in range(H):                                           # one step ahead, true history
            _, _, rr = model.rollout(o_n[:, k:k + M], act[:, k:], 1)
            tf.append(rs(rr.mean(0)[:, 0]))
        _, _, rr = model.rollout(o_n[:, :M], act, H)
        return torch.stack(tf, 1), rs(rr.mean(0))

    # --- policy in the real env
    O, A_, R = [], [], []
    for i in range(a.episodes):
        o, _ = env.reset(seed=i)
        obs, acts, rews = [o], [], []
        for t in range(M + H):
            x = torch.as_tensor(np.clip(norm.obs(o), -OBS_CLIP, OBS_CLIP), dtype=torch.float32)
            act = ac.pi(x).clamp(-1, 1).numpy()
            o, rw, term, trunc, _ = env.step(act)
            obs.append(o); acts.append(act); rews.append(rw)
            if term:
                break
        if len(rews) == M + H:
            O.append(obs); A_.append(acts); R.append(rews)
    out = {}
    if O:
        o_n = torch.as_tensor(np.clip(norm.obs(np.array(O)), -OBS_CLIP, OBS_CLIP), dtype=torch.float32)
        act = torch.as_tensor(np.array(A_), dtype=torch.float32)
        real = np.array(R)[:, M - 1:M - 1 + H]
        tf, ol = model_rewards(o_n, act)
        img = Imagination(model); img.reset(o_n[:, :M], act[:, :M - 1])
        cl, cur = [], img.cur
        for k in range(H):
            _, _, rr = img.step(ac.pi(cur).clamp(-1, 1)); cur = img.cur
            cl.append(rs(rr.mean(0)))
        out["policy"] = dict(n=len(O), real=float(real.mean()), teacher_forced=float(tf.mean()),
                             open_loop=float(ol.mean()), closed_loop_imagined=float(torch.stack(cl).mean()),
                             act_abs_mean=float(act.abs().mean()), act_saturated=float((act.abs() > 0.99).float().mean()))
    else:
        out["policy"] = "every episode terminated before hist+horizon steps"

    # --- dataset (behaviour policy) reference
    eps = load_source(wa["train"], None, seed=wa["seed"])
    _, te = split_episodes(eps, wa["test_frac"], seed=12345)
    W = WindowSet(te, norm, M, H, stride=97)
    ids = np.random.default_rng(0).choice(len(W), min(64, len(W)), replace=False)
    o_n, act, rr_n = (torch.from_numpy(x) for x in W.get(ids))
    real = rs(rr_n[:, M - 1:M - 1 + H])
    tf, ol = model_rewards(o_n, act)
    out["dataset"] = dict(n=len(ids), real=float(real.mean()), teacher_forced=float(tf.mean()),
                          open_loop=float(ol.mean()), act_abs_mean=float(act.abs().mean()),
                          act_saturated=float((act.abs() > 0.99).float().mean()))
    print(json.dumps(dict(run=Path(a.policy).name, **out), indent=1))


if __name__ == "__main__":
    main()
