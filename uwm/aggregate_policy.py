"""Collect policy result.json files into one table (mean +- std over seeds).

python -m uwm.aggregate_policy --runs $SCRATCH/uwm_policy --out policy_results.csv
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

KEYS = ["real_return", "real_len", "dataset_return", "img_reward_per_step", "real_reward_per_step",
        "exploitation_gap"]


def main(argv=None):
    p = argparse.ArgumentParser()
    p.add_argument("--runs", required=True)
    p.add_argument("--out", default="policy_results.csv")
    a = p.parse_args(argv)
    rows = []
    for f in Path(a.runs).glob("*/result.json"):
        r = json.loads(f.read_text())
        name = f.parent.name
        m = re.match(r"(halfcheetah|hopper|walker2d)_(shared_none|independent_none)_(.+)_s(\d+)$", name)
        if not m:
            continue
        fr = r["final_real"]
        rows.append(dict(env=m[1], wm=m[2], penalty=m[3], seed=int(m[4]), real_return=fr["ret_mean"],
                         real_len=fr["len_mean"], dataset_return=r["dataset_return"],
                         img_reward_per_step=r["final_img_reward_per_step"],
                         real_reward_per_step=fr["per_step"], exploitation_gap=r["exploitation_gap"]))
    df = pd.DataFrame(rows)
    df.to_csv(Path(a.out).with_suffix(".raw.csv"), index=False)
    g = df.groupby(["env", "wm", "penalty"])[KEYS].agg(["mean", "std"])
    g.to_csv(a.out)
    with pd.option_context("display.width", 250, "display.max_columns", 50, "display.max_rows", 200):
        print(g.xs("mean", axis=1, level=1).round(2))
        print(f"\n{len(df)} runs")


if __name__ == "__main__":
    main()
