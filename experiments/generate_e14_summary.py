"""
generate_e14_summary.py
===============================================================================
Reads the per-seed E14_operating_point_e12.csv and produces an aggregated
E14_operating_point_summary.csv (mean ± std) suitable for inclusion in
docs/RESEARCH_REPORT.md.

Run after E14_operating_point_e12.csv is available.
"""

import pandas as pd
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "data" / "results"
IN = RESULTS / "E14_operating_point_e12.csv"
OUT = RESULTS / "E14_operating_point_summary.csv"

df = pd.read_csv(IN)
rows = []
for (ds, emb), g in df.groupby(["dataset", "embedder"]):
    for m in ["A1_raw", "B1_raw", "B1b_SigmaT", "B1w_SigmaW"]:
        sub = g[g["method"] == m]
        rows.append({
            "dataset": ds,
            "embedder": emb,
            "method": m,
            "mean_roc_auc": float(sub["roc_auc"].mean()),
            "std_roc_auc": float(sub["roc_auc"].std(ddof=1)),
            "mean_tpr_fpr1": float(sub["tpr_fpr1"].mean()),
            "mean_tpr_fpr5": float(sub["tpr_fpr5"].mean()),
            "mean_tpr_fpr10": float(sub["tpr_fpr10"].mean()),
            "mean_fpr_tpr90": float(sub["fpr_tpr90"].mean()),
            "mean_brier": float(sub["brier"].mean()),
            "mean_ece10": float(sub["ece10"].mean()),
        })
out_df = pd.DataFrame(rows)
out_df.to_csv(OUT, index=False)
print(f"Written {len(out_df)} rows -> {OUT}")
print(out_df.round(4).to_string(index=False))
