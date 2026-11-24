"""Train one RWM-U style ensemble on an offline dataset.

python -m uwm.train --train minari:mujoco/halfcheetah/medium-v0 --trunk shared --bootstrap none \
    --members 5 --seed 0 --out runs/hc_shared_none_s0
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch

from .data import Normalizer, WindowSet, load_source, split_episodes
from .model import EnsembleWorldModel


def gaussian_nll(y, mu, logstd):
    return (0.5 * ((y - mu) / logstd.exp()) ** 2 + logstd).mean(-1)


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--train", required=True, help="dataset spec, see data.load_source")
    p.add_argument("--max-episodes", type=int, default=None)
    p.add_argument("--test-frac", type=float, default=0.1)
    p.add_argument("--members", type=int, default=5)
    p.add_argument("--trunk", choices=["shared", "independent"], default="shared")
    p.add_argument("--feedback", choices=["mean", "member"], default="mean")
    p.add_argument("--bootstrap", choices=["none", "poisson"], default="none",
                   help="poisson = online bootstrap: per-member Poisson(1) weight on every window")
    p.add_argument("--loss", choices=["nll", "mse"], default="nll")
    p.add_argument("--hist", type=int, default=32)
    p.add_argument("--horizon", type=int, default=8)
    p.add_argument("--iters", type=int, default=20000)
    p.add_argument("--batch", type=int, default=1024)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--wd", type=float, default=1e-5)
    p.add_argument("--reward-weight", type=float, default=1.0)
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--log-every", type=int, default=500)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    eps = load_source(a.train, a.max_episodes, seed=a.seed)
    # the episode split uses a fixed seed so every run of the grid sees the same held-out episodes
    train_eps, test_eps = split_episodes(eps, a.test_frac, seed=12345)
    norm = Normalizer(train_eps)
    W = WindowSet(train_eps, norm, a.hist, a.horizon)
    Wval = WindowSet(test_eps, norm, a.hist, a.horizon, stride=50)
    D, A = train_eps[0]["obs"].shape[1], train_eps[0]["act"].shape[1]

    model = EnsembleWorldModel(D, A, a.members, a.trunk, a.feedback).to(dev)
    opt = torch.optim.Adam(model.parameters(), lr=a.lr, weight_decay=a.wd)
    sched = torch.optim.lr_scheduler.CosineAnnealingLR(opt, a.iters)

    if a.bootstrap == "poisson":
        boot = torch.from_numpy(rng.poisson(1.0, (a.members, len(W))).astype(np.float32))
    print(f"[data] train eps={len(train_eps)} windows={len(W)}  val eps={len(test_eps)}  D={D} A={A}  dev={dev}")

    def batch_to(ids):
        o, ac, r = W.get(ids)
        return (torch.from_numpy(o).to(dev), torch.from_numpy(ac).to(dev), torch.from_numpy(r).to(dev))

    def loss_fn(o, ac, r, w=None):
        M, N = a.hist, a.horizon
        mu, ls, rp = model.rollout(o[:, :M], ac, N)                 # [E,B,N,D]
        y, ry = o[:, M:M + N], r[:, M - 1:M - 1 + N]
        if a.loss == "nll":
            lo = gaussian_nll(y, mu, ls)                               # [E,B,N]
        else:
            lo = ((y - mu) ** 2).mean(-1)
        lr_ = (rp - ry) ** 2
        per = (lo + a.reward_weight * lr_).mean(-1)                    # [E,B]
        if w is None:
            return per.mean(), lo.mean().item(), lr_.mean().item()
        return (per * w).sum() / w.sum().clamp_min(1.0), lo.mean().item(), lr_.mean().item()

    log, t0 = [], time.time()
    for it in range(1, a.iters + 1):
        ids = rng.integers(0, len(W), a.batch)
        o, ac, r = batch_to(ids)
        w = boot[:, ids].to(dev) if a.bootstrap == "poisson" else None
        loss, lo, lr_ = loss_fn(o, ac, r, w)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), 10.0)
        opt.step(); sched.step()
        if it % a.log_every == 0 or it == a.iters:
            model.eval()
            with torch.no_grad():
                vids = np.arange(min(len(Wval), 2048))
                vo, va, vr = (torch.from_numpy(x).to(dev) for x in Wval.get(vids))
                mu, _, _ = model.rollout(vo[:, :a.hist], va, a.horizon)
                vmse = ((mu.mean(0) - vo[:, a.hist:a.hist + a.horizon]) ** 2).mean().item()
            model.train()
            row = dict(it=it, loss=loss.item(), obs_loss=lo, rew_loss=lr_, val_mse=vmse, sec=time.time() - t0)
            log.append(row); print(json.dumps(row), flush=True)

    torch.save(dict(state=model.state_dict(), args=vars(a), D=D, A=A), out / "model.pt")
    with open(out / "norm.pkl", "wb") as f:
        pickle.dump(norm.state_dict(), f)
    with open(out / "split.pkl", "wb") as f:
        pickle.dump(dict(test_eps=test_eps), f)
    (out / "train_log.json").write_text(json.dumps(log))


if __name__ == "__main__":
    main()
