"""Stretch goal: MOPO-style PPO inside a trained world-model ensemble, with uncertainty penalties.

Question: given that the ensemble's uncertainty is overconfident in absolute terms but ranks
errors reasonably (Idea 1), which way of turning it into a reward penalty actually helps a policy
trained purely in imagination when it is deployed in the real MuJoCo env?

Reward used for PPO at each imagined step (per rollout b):
    r_train = r_hat / r_std  -  lam * u / u_ref          (--norm, default)
    r_train = r_hat / r_std  -  lam * u                  (--raw)
where r_hat is the ensemble-mean predicted reward in real units, r_std is the dataset reward std,
and u_ref is the mean of u over one-step predictions on training data. Normalising by u_ref puts
the three penalties on the same scale, so one lam means the same thing for each of them.

Penalties u (all in normalised observation units):
    epi   sum_d Var_e[mu_e,d]                  ensemble variance of the means (RWM-U, Eq. 4)
    ale   max_e ||sigma_e||_2                  MOPO's max aleatoric std (Yu et al. 2020)
    pair  max_{i,j} ||mu_i - mu_j||_2           max pairwise disagreement (MOReL-style)
    none  0                                    no penalty (lam ignored)

Imagined rollouts start from 32-step true histories sampled from the training split, run for
--horizon steps with the world model's own feedback mode, and stop early when the hand-coded
Hopper/Walker2d health check fails (the model does not predict termination). Policies are then
evaluated in the real environment.

python -m uwm.policy --wm $SCRATCH/uwm_runs/halfcheetah_shared_none_s0 \
    --penalty epi --lam 1 --seed 0 --out $SCRATCH/uwm_policy/hc_epi_s0
"""
from __future__ import annotations

import argparse
import json
import pickle
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

from .data import Normalizer, WindowSet, load_source, split_episodes
from .model import EnsembleWorldModel

OBS_CLIP = 10.0          # normalised-units clamp on imagined states (stops rare blow-ups)
PENALTIES = ("none", "epi", "ale", "pair")


# --------------------------------------------------------------------------- env helpers

def env_kind(train_spec: str) -> str:
    s = train_spec.lower()
    for k in ("halfcheetah", "hopper", "walker2d"):
        if k in s:
            return k
    raise ValueError(f"cannot tell env from {train_spec!r}")


def terminated(kind: str, obs: torch.Tensor) -> torch.Tensor:
    """Gymnasium v5 health checks on REAL-unit observations [B, D] -> bool [B]."""
    if kind == "halfcheetah":
        return torch.zeros(obs.shape[0], dtype=torch.bool, device=obs.device)
    z, ang = obs[:, 0], obs[:, 1]
    finite = torch.isfinite(obs).all(-1)
    if kind == "hopper":
        healthy = finite & (obs[:, 1:].abs() < 100).all(-1) & (z > 0.7) & (ang.abs() < 0.2)
    else:  # walker2d
        healthy = finite & (z > 0.8) & (z < 2.0) & (ang > -1.0) & (ang < 1.0)
    return ~healthy


def make_real_env(train_spec: str):
    import gymnasium as gym
    kind, rest = train_spec.split(":", 1)
    if kind == "minari":
        import minari
        try:
            return minari.load_dataset(rest).recover_environment(eval_env=True)
        except Exception as ex:  # older minari / no eval env stored
            print(f"[env] recover_environment failed ({ex}); falling back to gym.make", flush=True)
    name = {"halfcheetah": "HalfCheetah-v5", "hopper": "Hopper-v5", "walker2d": "Walker2d-v5"}
    return gym.make(name[env_kind(train_spec)])


# --------------------------------------------------------------------------- imagination

class Imagination:
    """Step-wise version of EnsembleWorldModel.rollout (identical maths; see test in __main__)."""

    def __init__(self, model: EnsembleWorldModel):
        self.m = model
        self.E = model.E
        self.single = model.trunk_mode == "shared" and model.feedback == "mean"

    @torch.no_grad()
    def reset(self, obs_hist: torch.Tensor, act_hist: torch.Tensor):
        """obs_hist [B, M, D]; act_hist [B, M-1, A] (actions between the history observations)."""
        M = obs_hist.shape[1]
        n = 1 if self.m.trunk_mode == "shared" else self.E
        if M > 1:
            x = torch.cat([obs_hist[:, :M - 1], act_hist], -1)
            hs = [self.m._trunk(e)(x)[1] for e in range(n)]
        else:
            hs = [None] * n
        if self.m.trunk_mode == "shared" and not self.single:
            hs = hs * self.E                 # members diverge after the first step
        self.h = hs
        self.cur = obs_hist[:, -1]
        self.cur_m = [self.cur] * self.E

    @torch.no_grad()
    def step(self, a: torch.Tensor):
        """a [B, A] -> mu, logstd [E, B, D], r [E, B] (normalised units)."""
        mean_fb = self.m.feedback == "mean"
        mus, lss, rs = [], [], []
        if self.single:
            z, self.h[0] = self.m.trunks[0](torch.cat([self.cur, a], -1).unsqueeze(1), self.h[0])
            z = z[:, 0]
        for e in range(self.E):
            base = self.cur if mean_fb else self.cur_m[e]
            if not self.single:
                ze, self.h[e] = self.m._trunk(e)(torch.cat([base, a], -1).unsqueeze(1), self.h[e])
                ze = ze[:, 0]
            else:
                ze = z
            d, ls, r = self.m.heads[e](ze)
            mus.append(base + d); lss.append(ls); rs.append(r)
        mu, ls, r = torch.stack(mus), torch.stack(lss), torch.stack(rs)
        self.cur = mu.mean(0).clamp(-OBS_CLIP, OBS_CLIP)
        self.cur_m = list(mu.clamp(-OBS_CLIP, OBS_CLIP))
        return mu, ls, r


def penalty(kind: str, mu: torch.Tensor, ls: torch.Tensor) -> torch.Tensor:
    """mu, ls [E, B, D] -> u [B]."""
    if kind == "none":
        return torch.zeros(mu.shape[1], device=mu.device)
    if kind == "epi":
        return mu.var(0, unbiased=False).sum(-1)
    if kind == "ale":
        return ls.exp().norm(dim=-1).max(0).values
    if kind == "pair":
        m = mu.transpose(0, 1)                               # [B, E, D]
        return torch.cdist(m, m).flatten(1).max(-1).values
    raise ValueError(kind)


# --------------------------------------------------------------------------- policy

def mlp(i, o, w=256):
    return nn.Sequential(nn.Linear(i, w), nn.Tanh(), nn.Linear(w, w), nn.Tanh(), nn.Linear(w, o))


class ActorCritic(nn.Module):
    def __init__(self, D, A):
        super().__init__()
        self.pi, self.v = mlp(D, A), mlp(D, 1)
        self.log_std = nn.Parameter(torch.full((A,), -0.5))
        nn.init.zeros_(self.pi[-1].bias); self.pi[-1].weight.data.mul_(0.01)

    def dist(self, o):
        return torch.distributions.Normal(self.pi(o), self.log_std.exp())

    def value(self, o):
        return self.v(o).squeeze(-1)


@torch.no_grad()
def evaluate_real(ac, env, norm, dev, episodes, seed):
    rets, lens = [], []
    for i in range(episodes):
        o, _ = env.reset(seed=seed * 1000 + i)
        R, T, done = 0.0, 0, False
        while not done:
            x = torch.as_tensor(np.clip(norm.obs(o), -OBS_CLIP, OBS_CLIP), dtype=torch.float32, device=dev)
            a = ac.pi(x).clamp(-1, 1).cpu().numpy()
            o, r, term, trunc, _ = env.step(a)
            R += float(r); T += 1; done = term or trunc
        rets.append(R); lens.append(T)
    return dict(ret_mean=float(np.mean(rets)), ret_std=float(np.std(rets)), len_mean=float(np.mean(lens)),
                rets=rets, per_step=float(np.sum(rets) / np.sum(lens)))


# --------------------------------------------------------------------------- main

def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--wm", required=True, help="world-model run dir (model.pt, norm.pkl)")
    p.add_argument("--penalty", choices=PENALTIES, default="epi")
    p.add_argument("--lam", type=float, default=1.0)
    p.add_argument("--raw", action="store_true", help="do not normalise u by its training-data mean")
    p.add_argument("--iters", type=int, default=1000)
    p.add_argument("--batch", type=int, default=2048, help="parallel imagined rollouts per iteration")
    p.add_argument("--horizon", type=int, default=50)
    p.add_argument("--gamma", type=float, default=0.99)
    p.add_argument("--gae", type=float, default=0.95)
    p.add_argument("--clip", type=float, default=0.2)
    p.add_argument("--epochs", type=int, default=5)
    p.add_argument("--minibatch", type=int, default=8192)
    p.add_argument("--lr", type=float, default=3e-4)
    p.add_argument("--eval-every", type=int, default=100)
    p.add_argument("--eval-episodes", type=int, default=5)
    p.add_argument("--final-episodes", type=int, default=10)
    p.add_argument("--max-episodes", type=int, default=None, help="cap dataset size (smoke tests)")
    p.add_argument("--seed", type=int, default=0)
    p.add_argument("--out", required=True)
    a = p.parse_args(argv)

    out = Path(a.out); out.mkdir(parents=True, exist_ok=True)
    torch.manual_seed(a.seed); rng = np.random.default_rng(a.seed)
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    ck = torch.load(Path(a.wm) / "model.pt", map_location=dev, weights_only=False)
    wa = ck["args"]
    model = EnsembleWorldModel(ck["D"], ck["A"], wa["members"], wa["trunk"], wa["feedback"]).to(dev)
    model.load_state_dict(ck["state"]); model.eval()
    for q in model.parameters():
        q.requires_grad_(False)
    with open(Path(a.wm) / "norm.pkl", "rb") as f:
        norm = Normalizer.from_state(pickle.load(f))
    kind = env_kind(wa["train"])
    M = wa["hist"]

    # same episode split as the world model saw (fixed seed in train.py)
    eps = load_source(wa["train"], a.max_episodes or wa.get("max_episodes"), seed=wa["seed"])
    train_eps, _ = split_episodes(eps, wa["test_frac"], seed=12345)
    W = WindowSet(train_eps, norm, M, 1)
    data_ret = float(np.mean([e["rew"].sum() for e in train_eps]))
    data_len = float(np.mean([len(e["rew"]) for e in train_eps]))
    r_mean = torch.tensor(norm.r_mean, device=dev); r_std = torch.tensor(norm.r_std, device=dev)
    o_mean = torch.as_tensor(norm.mean, device=dev); o_std = torch.as_tensor(norm.std, device=dev)
    img = Imagination(model)

    def starts(n):
        o, ac, _ = W.get(rng.integers(0, len(W), n))
        return torch.from_numpy(o[:, :M]).to(dev), torch.from_numpy(ac[:, :M - 1]).to(dev), \
            torch.from_numpy(ac[:, M - 1]).to(dev)

    # reference scale of each penalty: mean u for one-step predictions on training data
    u_ref = {}
    with torch.no_grad():
        oh, ah, a_data = starts(8192)
        img.reset(oh, ah)
        mu, ls, _ = img.step(a_data)
        for k in PENALTIES[1:]:
            u_ref[k] = float(penalty(k, mu, ls).mean())
    scale = 1.0 if (a.raw or a.penalty == "none") else 1.0 / u_ref[a.penalty]
    lam = 0.0 if a.penalty == "none" else a.lam
    print(f"[policy] env={kind} wm={a.wm} penalty={a.penalty} lam={lam} raw={a.raw} u_ref={u_ref} "
          f"dataset return={data_ret:.1f} (len {data_len:.0f}) dev={dev}", flush=True)

    D, A = ck["D"], ck["A"]
    ac = ActorCritic(D, A).to(dev)
    opt = torch.optim.Adam(ac.parameters(), lr=a.lr)
    env = make_real_env(wa["train"])
    log, evals, t0 = [], [], time.time()
    H, B = a.horizon, a.batch

    for it in range(1, a.iters + 1):
        # ---------------- imagined rollouts
        obs_b = torch.zeros(H, B, D, device=dev); act_b = torch.zeros(H, B, A, device=dev)
        logp_b = torch.zeros(H, B, device=dev); rew_b = torch.zeros(H, B, device=dev)
        val_b = torch.zeros(H + 1, B, device=dev); alive_b = torch.zeros(H, B, device=dev)
        done_b = torch.zeros(H, B, device=dev)
        r_model_sum = torch.zeros(B, device=dev); u_sum = 0.0; n_alive = 0.0; sat = 0.0; ue_sum = 0.0
        with torch.no_grad():
            oh, ah, _ = starts(B)
            img.reset(oh, ah)
            o = img.cur.clamp(-OBS_CLIP, OBS_CLIP)
            alive = torch.ones(B, device=dev)
            for t in range(H):
                d = ac.dist(o); act = d.sample()
                obs_b[t], act_b[t], logp_b[t] = o, act, d.log_prob(act).sum(-1)
                val_b[t] = ac.value(o); alive_b[t] = alive
                mu, ls, r = img.step(act.clamp(-1, 1))
                r_real = r.mean(0) * r_std + r_mean
                u = penalty(a.penalty, mu, ls)
                rew_b[t] = (r_real / r_std - lam * scale * u) * alive
                r_model_sum += r_real * alive
                u_sum += float((u * alive).sum()); n_alive += float(alive.sum())
                ue_sum += float((penalty("epi", mu, ls) * alive).sum())   # logged for every run
                sat += float(((ac.pi(o).abs() > 1).float().mean(-1) * alive).sum())
                term = terminated(kind, img.cur * o_std + o_mean).float()
                done_b[t] = term * alive
                alive = alive * (1 - term)
                o = img.cur
            val_b[H] = ac.value(o) * alive

            # GAE; steps after termination have alive=0 and are dropped from the loss
            adv = torch.zeros(H, B, device=dev); last = torch.zeros(B, device=dev)
            for t in reversed(range(H)):
                nonterm = 1 - done_b[t]
                delta = rew_b[t] + a.gamma * val_b[t + 1] * nonterm - val_b[t]
                last = delta + a.gamma * a.gae * nonterm * last
                adv[t] = last * alive_b[t]
            ret = adv + val_b[:H]

        # ---------------- PPO update
        m = alive_b.flatten() > 0
        O, Ac, LP, AD, RT = (x.flatten(0, 1)[m] for x in (obs_b, act_b, logp_b, adv, ret))
        AD = (AD - AD.mean()) / (AD.std() + 1e-8)
        N = O.shape[0]
        for _ in range(a.epochs):
            perm = torch.randperm(N, device=dev)
            for s in range(0, N, a.minibatch):
                idx = perm[s:s + a.minibatch]
                d = ac.dist(O[idx]); lp = d.log_prob(Ac[idx]).sum(-1)
                ratio = (lp - LP[idx]).exp()
                pl = -torch.min(ratio * AD[idx], ratio.clamp(1 - a.clip, 1 + a.clip) * AD[idx]).mean()
                vl = ((ac.value(O[idx]) - RT[idx]) ** 2).mean()
                opt.zero_grad(set_to_none=True)
                (pl + 0.5 * vl).backward()
                nn.utils.clip_grad_norm_(ac.parameters(), 1.0)
                opt.step()

        row = dict(it=it, img_reward_per_step=float(r_model_sum.sum() / n_alive),
                   img_alive_frac=float(alive_b.mean()), u_mean=u_sum / n_alive,
                   u_mean_rel=(u_sum / n_alive) / u_ref[a.penalty] if a.penalty != "none" else 0.0,
                   pi_std=float(ac.log_std.exp().mean()), act_saturated=sat / n_alive,
                   u_epi_rel=(ue_sum / n_alive) / u_ref["epi"], sec=time.time() - t0)
        if it % a.eval_every == 0 or it == a.iters:
            ev = evaluate_real(ac, env, norm, dev, a.final_episodes if it == a.iters else a.eval_episodes,
                               seed=10_000 + it)
            row.update(real_return=ev["ret_mean"], real_len=ev["len_mean"], real_reward_per_step=ev["per_step"])
            evals.append(dict(it=it, **ev))
        log.append(row)
        if it % 10 == 0 or "real_return" in row:
            print(json.dumps(row), flush=True)

    torch.save(dict(state=ac.state_dict(), args=vars(a), D=D, A=A), out / "policy.pt")
    (out / "policy_log.json").write_text(json.dumps(log))
    final = evals[-1]
    # u for the penalty kinds, measured on the final policy's imagined rollouts (diagnostic)
    result = dict(args=vars(a), wm_args=wa, env=kind, u_ref=u_ref, dataset_return=data_ret,
                  dataset_len=data_len, final_real=final, evals=evals,
                  final_img_reward_per_step=log[-1]["img_reward_per_step"], final_u_mean_rel=log[-1]["u_mean_rel"],
                  final_act_saturated=log[-1]["act_saturated"], final_u_epi_rel=log[-1]["u_epi_rel"],
                  exploitation_gap=log[-1]["img_reward_per_step"] - final["per_step"])
    (out / "result.json").write_text(json.dumps(result, indent=1))
    print(f"[done] real return {final['ret_mean']:.1f} ± {final['ret_std']:.1f} (dataset {data_ret:.1f}); "
          f"imagined/real reward per step {log[-1]['img_reward_per_step']:.3f}/{final['per_step']:.3f}",
          flush=True)


if __name__ == "__main__":
    main()
