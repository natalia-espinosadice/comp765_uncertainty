"""Collect eval_*.json files into one table (mean +- std over seeds).

python -m uwm.aggregate --runs $SCRATCH/uwm_runs --out results.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

KEYS = ["rmse@100", "spearman_pooled", "spearman_within_mean", "cov1", "cov2", "cov1_epi",
        "mean_abs_z", "auroc_u_sum@1", "auroc_u_sum@10", "auroc_u_sum@100", "auroc_oracle_err@100"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default="results.csv")
    a = p.parse_args(argv)

    rows = []
    for f in Path(a.runs).glob("*/eval_*.json"):
        r = json.loads(f.read_text())
        name = f.parent.name
        cfg = re.sub(r"_s\d+$", "", name)
        for test_set, m in r.items():
            if not isinstance(m, dict) or "n_windows" not in m:
                continue
            short = "id" if test_set == "id" else test_set.split("/")[-1].replace("-v0", "")
            rows.append(dict(config=cfg, seed=name.rsplit("_s", 1)[-1], eval_feedback=r["eval_feedback"],
                             test_set=short, **{k: m.get(k) for k in KEYS}))
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out).with_suffix(".raw.csv"), index=False)
    g = df.groupby(["config", "eval_feedback", "test_set"])[KEYS].agg(["mean", "std"])
    g.to_csv(a.out)
    with pd.option_context("display.width", 250, "display.max_columns", 50):
        print(g.xs("mean", axis=1, level=1).round(3))


if __name__ == "__main__":
    main()
