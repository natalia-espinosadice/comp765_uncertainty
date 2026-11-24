"""Offline datasets for the RWM-U uncertainty study.

Episodes are stored as dicts of numpy arrays: obs [T+1, D], act [T, A], rew [T].
Two sources:
  * Minari (`mujoco/<env>/<quality>-v0`), the main source on the cluster.
  * A local Gymnasium generator (random / noisy-sinusoid policies), used for smoke tests
    and as an extra, clearly out-of-distribution test set.
"""
from __future__ import annotations

import numpy as np


# --------------------------------------------------------------------------- loading

def load_minari(name: str, max_episodes: int | None = None) -> list[dict]:
    import minari

    ds = minari.load_dataset(name, download=True)
    eps = []
    for i, ep in enumerate(ds.iterate_episodes()):
        if max_episodes is not None and i >= max_episodes:
            break
        eps.append(dict(obs=np.asarray(ep.observations, np.float32),
                        act=np.asarray(ep.actions, np.float32),
                        rew=np.asarray(ep.rewards, np.float32)))
    return eps


def generate_gym(env_id: str, policy: str, n_episodes: int, ep_len: int = 1000,
                 seed: int = 0) -> list[dict]:
    """policy: 'random' (uniform actions) or 'sine' (noisy open-loop sinusoids, a crude gait)."""
    import gymnasium as gym

    env = gym.make(env_id, max_episode_steps=ep_len)
    rng = np.random.default_rng(seed)
    A = env.action_space.shape[0]
    eps = []
    for i in range(n_episodes):
        o, _ = env.reset(seed=seed * 10_000 + i)
        obs, act, rew = [o], [], []
        freq, phase = rng.uniform(1.0, 3.0), rng.uniform(0, 2 * np.pi, A)
        for t in range(ep_len):
            if policy == "random":
                a = rng.uniform(-1, 1, A)
            elif policy == "sine":
                a = np.sin(2 * np.pi * freq * t * env.unwrapped.dt + phase) + 0.2 * rng.standard_normal(A)
            else:
                raise ValueError(policy)
            a = np.clip(a, -1, 1).astype(np.float32)
            o, r, term, trunc, _ = env.step(a)
            obs.append(o); act.append(a); rew.append(r)
            if term or trunc:
                break
        eps.append(dict(obs=np.asarray(obs, np.float32), act=np.asarray(act, np.float32),
                        rew=np.asarray(rew, np.float32)))
    return eps


def load_source(spec: str, max_episodes: int | None = None, seed: int = 0) -> list[dict]:
    """spec examples: 'minari:mujoco/halfcheetah/medium-v0', 'gym:HalfCheetah-v5:random:20'."""
    kind, rest = spec.split(":", 1)
    if kind == "minari":
        return load_minari(rest, max_episodes)
    if kind == "gym":
        env_id, policy, n = rest.split(":")
        return generate_gym(env_id, policy, int(n), seed=seed)
    raise ValueError(spec)


def split_episodes(eps: list[dict], test_frac: float, seed: int) -> tuple[list[dict], list[dict]]:
    """Split by episode (never by window) so test windows are unseen trajectories."""
    idx = np.random.default_rng(seed).permutation(len(eps))
    n_test = max(1, int(round(test_frac * len(eps))))
    return [eps[i] for i in idx[n_test:]], [eps[i] for i in idx[:n_test]]


# --------------------------------------------------------------------------- normalisation

class Normalizer:
    def __init__(self, eps: list[dict]):
        o = np.concatenate([e["obs"] for e in eps])
        self.mean = o.mean(0)
        self.std = o.std(0) + 1e-6
        r = np.concatenate([e["rew"] for e in eps])
        self.r_mean, self.r_std = float(r.mean()), float(r.std() + 1e-6)

    def obs(self, x):
        return (x - self.mean) / self.std

    def state_dict(self):
        return dict(mean=self.mean, std=self.std, r_mean=self.r_mean, r_std=self.r_std)

    @classmethod
    def from_state(cls, s):
        n = cls.__new__(cls)
        n.mean, n.std, n.r_mean, n.r_std = s["mean"], s["std"], s["r_mean"], s["r_std"]
        return n


# --------------------------------------------------------------------------- windows

class WindowSet:
    """All (episode, start) windows of length hist + horizon, stored normalised.

    A window gives obs[start : start+hist+horizon+1] and act[start : start+hist+horizon].
    """

    def __init__(self, eps: list[dict], norm: Normalizer, hist: int, horizon: int, stride: int = 1):
        self.hist, self.horizon = hist, horizon
        self.obs = [norm.obs(e["obs"]).astype(np.float32) for e in eps]
        self.act = [e["act"] for e in eps]
        self.rew = [((e["rew"] - norm.r_mean) / norm.r_std).astype(np.float32) for e in eps]
        L = hist + horizon
        index = []
        for i, e in enumerate(eps):
            T = len(e["act"])
            for s in range(0, T - L + 1, stride):
                index.append((i, s))
        if not index:
            raise ValueError(f"no episode is long enough for hist+horizon={L}")
        self.index = np.asarray(index, np.int64)

    def __len__(self):
        return len(self.index)

    def get(self, ids: np.ndarray):
        L = self.hist + self.horizon
        o = np.stack([self.obs[e][s:s + L + 1] for e, s in self.index[ids]])
        a = np.stack([self.act[e][s:s + L] for e, s in self.index[ids]])
        r = np.stack([self.rew[e][s:s + L] for e, s in self.index[ids]])
        return o, a, r
