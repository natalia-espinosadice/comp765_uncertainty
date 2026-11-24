# Why is RWM-U's ensemble overconfident? (COMP 765 final project code)

## Hypothesis
RWM-U (Li, Krause, Hutter, arXiv 2504.16680) only ensembles the output heads on top of one shared GRU
(Sec. 4.1, Table S7). We test whether sharing the trunk, and training every member on the same data,
limits member diversity enough to make the epistemic estimate overconfident. That matches the
overconfidence that arXiv 2611.02699 measured on the released ANYmal checkpoint
(±1σ coverage 4.6% at h=100).

**H1.** Independent trunks and/or a data bootstrap raise coverage toward nominal and improve
within-horizon error ranking, at equal or better RMSE.
**H2.** Fig. 3's "uncertainty tracks error" is mostly driven by both quantities growing with the
horizon. At a fixed horizon step, the rank correlation across rollouts is much weaker.
**H3.** Uncertainty detects shifted data (train on `medium`, test on `simple`/`expert`) only
partially, and less well than the true error would.

## Setup
- **Data:** Minari `mujoco/{halfcheetah,hopper,walker2d}/{simple,medium,expert}-v0`. Train on
  `medium` (90% of episodes). Test in-distribution on the held-out 10% and out-of-distribution on
  `simple` and `expert`.
- **Model:** GRU 256x2 with 128-unit heads, one per member. Each head predicts a Gaussian (Δobs,
  log σ) plus a reward. History M=32 and training forecast N=8 (Table S10), trained with
  backpropagation through the autoregressive feedback.
- **Ablation:** trunk ∈ {shared, independent} × bootstrap ∈ {none, Poisson(1) online bootstrap},
  3 envs × 3 seeds. On HalfCheetah only, we also vary ensemble size (2, 10), rollout feedback
  (ensemble mean vs. each member on its own predictions) and the loss (NLL vs. MSE).
- **Evaluation:** 32 steps of true history, then a 100-step open-loop rollout with logged actions.
  For the metrics, see the `uwm/evaluate.py` docstring.

## Deviations from or assumptions about the paper (state these in the report)
1. Eq. 4 does not say how the variance is aggregated over observation dimensions. We work in
   per-dim normalised units and report both the sum (main metric) and the max.
2. The paper says it uses "bootstrap ensembles" but does not describe any data resampling. We test
   both with and without it.
3. Eq. 4 feeds the ensemble mean back into the model; we treat that as the default (`--feedback mean`).
4. Loss: Gaussian NLL with a soft-bounded log σ. 2611.02699 argues the original setup drives σ→0;
   the `--loss mse` run checks that.
5. Optimiser: Adam with lr 3e-4 and cosine decay over 20k iterations (paper: lr 1e-4, 2500
   iterations, on 6M transitions). Check `train_log.json` for convergence.
6. No privileged-information heads (contacts); MuJoCo has no equivalent in the offline datasets.

## Running on Mila
```bash
bash slurm/setup_mila.sh                 # login node, once: venv + dataset download + grid
mkdir -p logs
sbatch --array=0-56%20 slurm/run_grid.sbatch
python -m uwm.aggregate --runs $SCRATCH/uwm_runs --out results.csv
```
Each job trains one ensemble and then evaluates it under both feedback modes. Each run writes
`model.pt`, `train_log.json`, `eval_{mean,member}.json` and per-horizon curves in
`eval_*_curves.npz` for plotting.

The local smoke test used generated data (`--train gym:HalfCheetah-v5:sine:12`), because
Hugging Face is unreachable from the build machine. The Minari dataset names have not been verified
yet; `setup_mila.sh` prints them as it downloads.

## Stretch (Idea 2, only if Idea 1 models are done by ~Nov 29)
MOPO-PPO in imagination using the reward head. Compare penalties: Var of means (paper), MOPO's max
aleatoric σ, and max pairwise disagreement, each raw vs. normalised by its value on the training
data. Evaluate the policies in the real MuJoCo env. Not implemented yet.
