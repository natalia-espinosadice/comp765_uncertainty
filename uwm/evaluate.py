"""Evaluate uncertainty quality of a trained ensemble over long open-loop rollouts.

For each test set, windows of 32 true history steps are followed by an H-step autoregressive
rollout driven by the logged actions (as in the paper's Fig. 3, but over many windows).

Metrics (all in normalised observation units, which sidesteps the paper's unstated
aggregation over dimensions with very different physical scales):
  rmse[k]                 error of the ensemble mean at horizon step k
  spearman_pooled         rank corr. of u (sum of per-dim epistemic var) vs squared error over
                          all (window, step) pairs. Inflated because both grow with k.
  spearman_within[k]      the same, computed across windows at a fixed k. This is the honest test
                          of "u tells you WHICH rollouts are wrong", which Fig. 3 cannot show.
  cov1/cov2 (total|epi)   fraction of per-dim errors inside +-1/2 sigma (ideal 0.683 / 0.954)
  mean_abs_z              E|error/sigma| (ideal sqrt(2/pi) = 0.798)
  auroc_*                 ID-vs-shifted detection from the mean u over the first k steps

python -m uwm.evaluate --run runs/hc_shared_none_s0 \
    --ood minari:mujoco/halfcheetah/expert-v0 minari:mujoco/halfcheetah/simple-v0
"""
from __future__ import annotations

import argparse
import json
import pickle
from pathlib import Path

import numpy as np
import torch
from scipy.stats import spearmanr
from sklearn.metrics import roc_auc_score

from .data import Normalizer, WindowSet, load_source
from .model import EnsembleWorldModel, uncertainty

AUROC_KS = (1, 10, 32, 100)


@torch.no_grad()
def rollout_stats(model, eps, norm, hist, H, stride, max_windows, dev, feedback, bs=512):
    W = WindowSet(eps, norm, hist, H, stride=stride)
    ids = np.arange(len(W))
    if len(ids) > max_windows:
        ids = np.random.default_rng(0).choice(ids, max_windows, replace=False)
    err2, u_sum, u_max, z, z_epi = [], [], [], [], []
    for s in range(0, len(ids), bs):
        o, a, _ = (torch.from_numpy(x).to(dev) for x in W.get(ids[s:s + bs]))
        mu, ls, _ = model.rollout(o[:, :hist], a, H, feedback=feedback)
        y = o[:, hist:hist + H]
        m = mu.mean(0)
        epi, ale = uncertainty(mu, ls)                          # [B,H,D]
        e = y - m
        err2.append((e ** 2).mean(-1).cpu())
        u_sum.append(epi.sum(-1).cpu()); u_max.append(epi.max(-1).values.cpu())
        z.append((e / (epi + ale).sqrt()).cpu()); z_epi.append((e / epi.clamp_min(1e-12).sqrt()).cpu())
    cat = lambda xs: torch.cat(xs).numpy()
    return dict(err2=cat(err2), u_sum=cat(u_sum), u_max=cat(u_max), z=cat(z), z_epi=cat(z_epi))


def summarize(s):
    err2, u = s["err2"], s["u_sum"]
    H = err2.shape[1]
    within = np.array([spearmanr(u[:, k], err2[:, k])[0] for k in range(H)])
    az, aze = np.abs(s["z"]), np.abs(s["z_epi"])
    per_k = dict(
        rmse=np.sqrt(err2.mean(0)), u_sum=u.mean(0), spearman_within=within,
        cov1=(az <= 1).mean((0, 2)), cov2=(az <= 2).mean((0, 2)),
        cov1_epi=(aze <= 1).mean((0, 2)), cov2_epi=(aze <= 2).mean((0, 2)),
        mean_abs_z=az.mean((0, 2)),
    )
    pick = [k for k in (0, 9, 31, 99) if k < H]
    scalar = dict(
        n_windows=int(err2.shape[0]),
        spearman_pooled=float(spearmanr(u.ravel(), err2.ravel())[0]),
        spearman_pooled_umax=float(spearmanr(s["u_max"].ravel(), err2.ravel())[0]),
        spearman_within_mean=float(np.nanmean(within)),
        cov1=float((az <= 1).mean()), cov2=float((az <= 2).mean()),
        cov1_epi=float((aze <= 1).mean()), cov2_epi=float((aze <= 2).mean()),
        mean_abs_z=float(az.mean()),
        **{f"{n}@{k + 1}": float(per_k[n][k]) for n in ("rmse", "spearman_within", "cov1", "cov2") for k in pick},
    )
    return scalar, per_k


def auroc(id_s, ood_s, key):
    out = {}
    for k in AUROC_KS:
        k = min(k, id_s[key].shape[1])
        a = id_s[key][:, :k].mean(1); b = ood_s[key][:, :k].mean(1)
        y = np.r_[np.zeros(len(a)), np.ones(len(b))]
        out[f"auroc_{key}@{k}"] = float(roc_auc_score(y, np.r_[a, b]))
    return out


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--run", required=True)
    p.add_argument("--ood", nargs="*", default=[], help="dataset specs treated as shifted test sets")
    p.add_argument("--ood-max-episodes", type=int, default=50)
    p.add_argument("--horizon", type=int, default=100)
    p.add_argument("--stride", type=int, default=25)
    p.add_argument("--max-windows", type=int, default=4000)
    p.add_argument("--feedback", choices=["mean", "member"], default=None,
                   help="override the rollout feedback used at training time")
    a = p.parse_args(argv)

    run = Path(a.run)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(run / "model.pt", map_location=dev, weights_only=False)
    ta = ck["args"]
    model = EnsembleWorldModel(ck["D"], ck["A"], ta["members"], ta["trunk"], ta["feedback"]).to(dev)
    model.load_state_dict(ck["state"]); model.eval()
    norm = Normalizer.from_state(pickle.load(open(run / "norm.pkl", "rb")))
    test_eps = pickle.load(open(run / "split.pkl", "rb"))["test_eps"]
    fb = a.feedback or ta["feedback"]

    sets = {"id": test_eps}
    for spec in a.ood:
        sets[spec] = load_source(spec, a.ood_max_episodes, seed=777)

    stats, results, curves = {}, {"run": str(run), "train_args": ta, "eval_feedback": fb}, {}
    for name, eps in sets.items():
        stats[name] = rollout_stats(model, eps, norm, ta["hist"], a.horizon, a.stride,
                                    a.max_windows, dev, fb)
        scalar, per_k = summarize(stats[name])
        if name != "id":
            scalar.update(auroc(stats["id"], stats[name], "u_sum"))
            scalar.update({k.replace("err2", "oracle_err"): v
                           for k, v in auroc(stats["id"], stats[name], "err2").items()})
        results[name] = scalar
        for k, v in per_k.items():
            curves[f"{name}|{k}"] = v
        print(name, json.dumps(scalar, indent=1))

    tag = f"eval_{fb}"
    (run / f"{tag}.json").write_text(json.dumps(results, indent=1))
    np.savez(run / f"{tag}_curves.npz", **curves)


if __name__ == "__main__":
    main()
