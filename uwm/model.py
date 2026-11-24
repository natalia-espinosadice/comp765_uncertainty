"""RWM-U style autoregressive GRU ensemble.

Paper (Li et al., Sec. 4.1 / Table S7): one shared GRU (256, 256) feature extractor, B MLP heads
(128, ReLU), each predicting a Gaussian over the next observation. Epistemic u = Var_b[mu_b].

Ablation knobs studied here:
  trunk      'shared' (paper) | 'independent' (one GRU per member, classic deep ensemble)
  feedback   'mean'   (paper Eq. 4: the ensemble mean is fed back) |
             'member' (each member rolls out on its own predictions, as in PETS-style sampling)

Convention: the GRU input at step t is [o_t, a_t]; its output predicts o_{t+1} (as o_t + delta)
and r_t. All quantities are in normalised units.
"""
from __future__ import annotations

import torch
import torch.nn as nn

LOGSTD_MIN, LOGSTD_MAX = -6.0, 1.0

# Safety net: if the installed cuDNN refuses this GPU, fall back to PyTorch's native GRU kernels
# (slower, numerically equivalent). With the cu126 wheel from setup_mila.sh this should not fire.
if torch.cuda.is_available():
    try:
        torch.backends.cudnn.is_acceptable(torch.zeros(1, device="cuda"))
    except RuntimeError as _ex:
        torch.backends.cudnn.enabled = False
        print(f"[model] cuDNN unusable on {torch.cuda.get_device_name()}, disabled: {_ex}", flush=True)


class Head(nn.Module):
    def __init__(self, h: int, d_obs: int, width: int = 128):
        super().__init__()
        self.net = nn.Sequential(nn.Linear(h, width), nn.ReLU(), nn.Linear(width, 2 * d_obs + 1))
        self.d = d_obs

    def forward(self, z):
        out = self.net(z)
        delta, logstd, r = out[..., :self.d], out[..., self.d:2 * self.d], out[..., -1]
        # soft clamp keeps gradients alive at the bounds
        logstd = LOGSTD_MIN + (LOGSTD_MAX - LOGSTD_MIN) * torch.sigmoid(logstd)
        return delta, logstd, r


class EnsembleWorldModel(nn.Module):
    def __init__(self, d_obs: int, d_act: int, n_members: int = 5, trunk: str = "shared",
                 feedback: str = "mean", hidden: int = 256, layers: int = 2, head_width: int = 128):
        super().__init__()
        assert trunk in ("shared", "independent") and feedback in ("mean", "member")
        self.E, self.trunk_mode, self.feedback = n_members, trunk, feedback
        n_trunks = 1 if trunk == "shared" else n_members
        self.trunks = nn.ModuleList(
            nn.GRU(d_obs + d_act, hidden, num_layers=layers, batch_first=True) for _ in range(n_trunks))
        self.heads = nn.ModuleList(Head(hidden, d_obs, head_width) for _ in range(n_members))

    def _trunk(self, e):
        return self.trunks[0 if self.trunk_mode == "shared" else e]

    def rollout(self, obs_hist: torch.Tensor, act: torch.Tensor, horizon: int,
                feedback: str | None = None):
        """obs_hist [B, M, D] (true history o_0..o_{M-1}); act [B, >= M+horizon-1, A].

        Returns mu, logstd [E, B, horizon, D] for o_M..o_{M+horizon-1}, and r [E, B, horizon].
        """
        feedback = feedback or self.feedback
        B, M, D = obs_hist.shape
        x_hist = torch.cat([obs_hist, act[:, :M]], -1)

        # history pass: once for the shared trunk, once per trunk otherwise
        hs, zs = [], []
        for e in range(self.E):
            if self.trunk_mode == "shared" and e > 0:
                hs.append(hs[0]); zs.append(zs[0]); continue
            z, h = self._trunk(e)(x_hist)
            hs.append(h); zs.append(z[:, -1])

        prev = obs_hist[:, -1]                       # o_{M-1}
        prev_m = [prev] * self.E
        mus, lss, rs = [], [], []
        for k in range(horizon):
            step_mu, step_ls, step_r = [], [], []
            for e in range(self.E):
                if k > 0:
                    inp = (prev if feedback == "mean" else prev_m[e])
                    x = torch.cat([inp, act[:, M - 1 + k]], -1).unsqueeze(1)
                    z, hs[e] = self._trunk(e)(x, hs[e])
                    zs[e] = z[:, 0]
                base = prev if feedback == "mean" else prev_m[e]
                d, ls, r = self.heads[e](zs[e])
                step_mu.append(base + d); step_ls.append(ls); step_r.append(r)
            mu = torch.stack(step_mu)                # [E, B, D]
            mus.append(mu); lss.append(torch.stack(step_ls)); rs.append(torch.stack(step_r))
            prev, prev_m = mu.mean(0), list(mu)
        return torch.stack(mus, 2), torch.stack(lss, 2), torch.stack(rs, 2)


def uncertainty(mu: torch.Tensor, logstd: torch.Tensor):
    """mu, logstd [E, ..., D] -> per-dimension epistemic and aleatoric variance [..., D]."""
    epi = mu.var(0, unbiased=False)
    ale = (2 * logstd).exp().mean(0)
    return epi, ale
