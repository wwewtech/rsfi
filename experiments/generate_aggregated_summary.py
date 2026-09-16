"""
generate_aggregated_summary.py
================================================================================
Aggregated statistics across ALL committed per-seed experiment CSVs
(need.md task "Aggregated results"):

  1. mean-AUC summary + 95% confidence intervals recomputed from the
     per-seed roc_auc values (t-distribution, n-1 degrees of freedom);
  2. aggregated paired DeLong tests (wins / mean AUC diff / p-value summary).

Everything here is deterministic and reads ONLY committed CSVs from
data/results/ (no re-computation, no GPU).

Inputs (data/results/):
  per-seed AUC : E2d_safe_aware_multidataset.csv, E2q_qwen_multidataset.csv,
                 E8_sigma_w.csv, E8q_qwen_sigma_w.csv,
                 E12_advbench_harmbench_e2d.csv,
                 E13_cross_domain_transfer.csv, E14_operating_point_e12.csv,
                 E15_defense_aware_e12.csv,
                 E16_cross_domain_battery.csv   (optional: skipped if absent),
                 E13b_transfer_to_bigbench.csv  (optional: skipped if absent)
  DeLong       : E2d_delong_tests.csv, E2q_qwen_delong_tests.csv,
                 E8_delong_tests.csv, E8q_qwen_delong_tests.csv,
                 E12_advbench_harmbench_delong.csv,
                 E12_advbench_harmbench_delong_sigma_w.csv,
                 E16_cross_domain_delong.csv    (optional: skipped if absent)

Outputs (data/results/):
  AGGREGATED_mean_auc_ci.csv - per group: n_seeds, mean, std, ci_low, ci_high
  AGGREGATED_delong.csv     - per (dataset, model, pair)
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats

RESULTS = Path(__file__).parent.parent / "data" / "results"


def load(name: str) -> pd.DataFrame:
    return pd.read_csv(RESULTS / name)


def t_critical(n: int) -> float:
    """Two-sided 95% t critical value with n-1 degrees of freedom."""
    if n < 2:
        return np.nan
    return float(stats.t.ppf(0.975, n - 1))


def mean_auc_ci(df: pd.DataFrame, keys, source: str) -> pd.DataFrame:
    """Per key-group: n_seeds / mean / std / 95% CI of per-seed roc_auc.

    The t-interval is truncated to the [0, 1] support of the AUC: with few
    seeds and large cross-domain variance (E13b cells, std up to ~0.14) the
    raw interval can poke far past 1.0, which is not a meaningful AUC value
    (same bounded-support rationale as the Wilson CIs used for the rates in
    E17/E17b/E17c).
    """
    keys_list = keys if isinstance(keys, (list, tuple)) else [keys]
    rows = []
    for group_vals, g in df.groupby(keys_list):
        v = g["roc_auc"].to_numpy(dtype=float)
        n = int(len(v))
        m = float(v.mean())
        s = float(v.std(ddof=1)) if n >= 2 else np.nan
        ci = (t_critical(n) * s / np.sqrt(n)) if (n >= 2 and s == s) else np.nan
        vals = group_vals if isinstance(group_vals, tuple) else (group_vals,)
        rec = dict(zip(keys_list, vals))
        rec.update(source=source, n_seeds=n, mean_auc=m, std_auc=s,
                   ci95_low=float(min(max(m - ci, 0.0), 1.0)),
                   ci95_high=float(min(max(m + ci, 0.0), 1.0)))
        rows.append(rec)
    return pd.DataFrame(rows)


DELONG_COLS = ["dataset", "model", "seed", "pair",
               "method_1", "method_2", "auc_diff", "p_value"]


def aggregate_delong(df: pd.DataFrame, source: str) -> pd.DataFrame:
    """Group one per-seed DeLong CSV by (dataset, model, pair).

    Most DeLong CSVs use DELONG_COLS (dataset, model, pair, auc_diff,
    p_value). E13/E13b store the p-value in roc_auc without auc_diff and
    are aggregated separately by aggregate_delong_pvalue; files without an
    auc_diff column yield an empty frame here.
    """
    keep = [c for c in DELONG_COLS if c in df.columns]
    d = df[keep].copy()
    if d.empty or "auc_diff" not in d.columns:
        return pd.DataFrame()
    rows = []
    for (ds, mdl, pair), g in d.groupby(["dataset", "model", "pair"]):
        diffs = g["auc_diff"].to_numpy(dtype=float)
        p = g["p_value"].to_numpy(dtype=float)
        rows.append({
            "source": source, "dataset": ds, "model": mdl, "pair": pair,
            "n_seeds": int(len(g)),
            "auc_diff_mean": float(diffs.mean()),
            "auc_diff_std": float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan,
            "wins_m1": int((diffs > 0).sum()),
            "ties": int((diffs == 0).sum()),
            "losses": int((diffs < 0).sum()),
            "p_mean": float(p.mean()),
            "p_max": float(p.max()),
            "frac_p_below_0_05": float((p < 0.05).mean()),
        })
    return pd.DataFrame(rows)
def aggregate_delong_pvalue(df: pd.DataFrame, source: str, pair: str) -> pd.DataFrame:
    """Aggregate E13/E13b-style rows where the p-value lives in roc_auc.

    Per (train_ds, target_ds, embedder): wins/ties/losses are counted from
    the per-seed AUCs of the compared methods (B1w_SigmaW vs A1_raw), NOT
    from the p-values - a significant p only means "different". DeLong
    p-values that are undefined (NaN under perfect separation) are excluded
    from p_mean/p_max/frac_p_below_0_05; n_valid_p reports how many seeds
    had a defined p-value. `pair` names the compared methods.
    """
    d = df[df.method == "DELONG_B1w_vs_A1"]
    if d.empty:
        return pd.DataFrame()
    a1 = df[df.method == "A1_raw"].set_index(
        ["train_ds", "target_ds", "embedder", "seed"]).roc_auc
    b1w = df[df.method == "B1w_SigmaW"].set_index(
        ["train_ds", "target_ds", "embedder", "seed"]).roc_auc
    joined = pd.DataFrame({"a1": a1, "b1w": b1w}).join(
        d.set_index(["train_ds", "target_ds", "embedder", "seed"]).roc_auc
        .rename("p")).dropna(subset=["a1", "b1w"]).reset_index()
    rows = []
    for (tr, tg, mdl), g in joined.groupby(["train_ds", "target_ds",
                                            "embedder"]):
        p = g["p"].to_numpy(dtype=float)
        valid = p[np.isfinite(p)]
        diff = (g["b1w"] - g["a1"]).to_numpy(dtype=float)
        rows.append({
            "source": source, "dataset": f"{tr}->{tg}", "model": mdl,
            "pair": pair, "n_seeds": int(len(g)),
            "auc_diff_mean": float(diff.mean()) if len(diff) else np.nan,
            "auc_diff_std": (float(diff.std(ddof=1)) if len(diff) > 1
                             else np.nan),
            "wins_m1": int((diff > 0).sum()),
            "ties": int((diff == 0).sum()),
            "losses": int((diff < 0).sum()),
            "n_valid_p": int(len(valid)),
            "p_mean": float(valid.mean()) if len(valid) else np.nan,
            "p_max": float(valid.max()) if len(valid) else np.nan,
            "frac_p_below_0_05": (float((valid < 0.05).mean())
                                  if len(valid) else np.nan),
        })
    return pd.DataFrame(rows)


def main():
    print("=" * 100)
    print("AGGREGATED STATISTICS (need.md: summary tables, aggregated "
          "DeLong, 95% CI for mean AUC)")
    print("=" * 100)

    # ---------------- 1. per-seed AUC blocks ------------------------------
    e2d = load("E2d_safe_aware_multidataset.csv")
    e2q = load("E2q_qwen_multidataset.csv")
    e8 = load("E8_sigma_w.csv")
    e8q = load("E8q_qwen_sigma_w.csv")
    e12 = load("E12_advbench_harmbench_e2d.csv")
    e13 = load("E13_cross_domain_transfer.csv")
    e14 = load("E14_operating_point_e12.csv")

    ci_blocks = []
    ci_blocks.append(mean_auc_ci(pd.concat([e2d, e2q], ignore_index=True),
                                 ["dataset", "model", "method"], "E2d/E2q"))
    ci_blocks.append(mean_auc_ci(pd.concat([e8, e8q], ignore_index=True),
                                 ["dataset", "model", "method"], "E8/E8q"))
    ci_blocks.append(mean_auc_ci(e12, ["dataset", "model", "method"], "E12"))
    ci_blocks.append(mean_auc_ci(e14, ["dataset", "embedder", "method"], "E14"))

    # NEED.md tasks 1-2 (2026-09): E15 defense-aware on E12 data and E16
    # cross-domain BigBench battery, integrated when their CSVs exist.
    try:
        e15 = load("E15_defense_aware_e12.csv")
        ci_blocks.append(mean_auc_ci(e15, ["dataset", "model",
                                           "evaluated_method"], "E15"))
    except FileNotFoundError:
        print("[skip] E15_defense_aware_e12.csv not found - run E15 first")
    try:
        e16 = load("E16_cross_domain_battery.csv")
        ci_blocks.append(mean_auc_ci(e16, ["dataset", "model", "method"], "E16"))
    except FileNotFoundError:
        print("[skip] E16_cross_domain_battery.csv not found - run E16 first")

    # E13 transfer: score rows only (DELONG_B1w_vs_A1 stores a p-value)
    e13s = e13[e13.method != "DELONG_B1w_vs_A1"].copy()
    e13s["dataset"] = e13s["train_ds"] + "->" + e13s["target_ds"]
    ci_blocks.append(mean_auc_ci(e13s, ["dataset", "embedder", "method"],
                                 "E13_transfer"))
    # E13b transfer into the BigBench targets: same E13 schema, composite
    # train->target key like the E13_transfer block above.
    try:
        e13b = load("E13b_transfer_to_bigbench.csv")
        e13b_s = e13b[e13b.method != "DELONG_B1w_vs_A1"].copy()
        e13b_s["dataset"] = e13b_s["train_ds"] + "->" + e13b_s["target_ds"]
        ci_blocks.append(mean_auc_ci(e13b_s, ["dataset", "embedder", "method"],
                                     "E13b_transfer_BB"))
    except FileNotFoundError:
        print("[skip] E13b_transfer_to_bigbench.csv not found - run E13b first")
    xd = e13s[e13s["train_ds"] != e13s["target_ds"]]
    ci_blocks.append(mean_auc_ci(xd, ["method"], "E13_crossdomain"))
    print("\n--- E13 cross-domain (train != target), mean AUC over all "
          "pairs x embedders x seeds ---")
    print(ci_blocks[-1].round(4).to_string(index=False))

    ci_all = pd.concat(ci_blocks, ignore_index=True)
    ci_all.to_csv(RESULTS / "AGGREGATED_mean_auc_ci.csv", index=False)
    print(f"\nSaved {len(ci_all)} rows -> "
          f"data/results/AGGREGATED_mean_auc_ci.csv")

    # ---------------- 2. aggregated DeLong --------------------------------
    delong_blocks = [
        aggregate_delong(load("E2d_delong_tests.csv"), "E2d"),
        aggregate_delong(load("E2q_qwen_delong_tests.csv"), "E2q"),
        aggregate_delong(load("E8_delong_tests.csv"), "E8"),
        aggregate_delong(load("E8q_qwen_delong_tests.csv"), "E8q"),
        aggregate_delong(load("E12_advbench_harmbench_delong.csv"), "E12"),
        aggregate_delong(load("E12_advbench_harmbench_delong_sigma_w.csv"),
                         "E12sigmaW"),
    ]
    try:
        delong_blocks.append(
            aggregate_delong(load("E16_cross_domain_delong.csv"), "E16"))
    except FileNotFoundError:
        print("[skip] E16_cross_domain_delong.csv not found - run E16 first")
    # E13/E13b store the DeLong p-value in roc_auc (no auc_diff column), so
    # they go through the dedicated p-value aggregator.
    for name in ["E13_cross_domain_transfer.csv",
                 "E13b_transfer_to_bigbench.csv"]:
        try:
            src = "E13" if name.startswith("E13_") else "E13b"
            delong_blocks.append(
                aggregate_delong_pvalue(load(name), src, "B1w_vs_A1"))
        except FileNotFoundError:
            print(f"[skip] {name} not found - run the experiment first")
    delong_all = pd.concat([b for b in delong_blocks if not b.empty],
                           ignore_index=True)
    delong_all.to_csv(RESULTS / "AGGREGATED_delong.csv", index=False)
    print(f"\nSaved {len(delong_all)} rows -> "
          f"data/results/AGGREGATED_delong.csv")

    # ---------------- 3. printed highlights for the report ----------------
    print("\n--- Headline aggregated AUC (mean | std | 95% CI) "
          "E2d/E2q (standard models + Qwen3-8B) ---")
    head_models = ["B1_discriminant_mean_raw", "A1_naive_cosine_raw",
                   "B1b_discriminant_mean_whitened"]
    with pd.option_context("display.width", 220, "display.max_columns", None):
        sub = ci_blocks[0][ci_blocks[0].method.isin(head_models)]
        print(sub[["dataset", "model", "method", "mean_auc", "std_auc",
                   "ci95_low", "ci95_high"]].round(4).to_string(index=False))
        print("\nE8/E8q (Sigma_W block):")
        print(ci_blocks[1].round(4).to_string(index=False))

    print("\n--- Aggregated DeLong: key pairs ---")
    pairs = ["B1_vs_A1", "B1b_vs_B1", "B1w_vs_B1b", "B1w_vs_B1",
             "B1b_vs_C1b", "B1_vs_C1"]
    dsub = delong_all[delong_all.pair.isin(pairs)]
    with pd.option_context("display.width", 240, "display.max_columns", None):
        print(dsub[["source", "dataset", "model", "pair", "n_seeds",
                    "auc_diff_mean", "wins_m1", "p_mean", "p_max",
                    "frac_p_below_0_05"]].round(4).to_string(index=False))


if __name__ == "__main__":
    main()