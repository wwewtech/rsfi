"""
E14_operating_point_e12.py
================================================================================
Operating-point and calibration audit on the 4th-block data (AdvBench /
HarmBench). AUC alone does not reflect production guardrail behavior; this
script reports per (dataset, embedder, method, seed):
  - TPR at FPR=1% / 5% / 10%  (strict operating points)
  - Brier score and ECE (10-bin expected calibration error) for the
    calibrated score s = (x - mu)^T d  (unnormalized projection), using
    isotonic-free binning on the REFERENCE pools only (no test leakage).
  - FPR at TPR=90% (false-positive budget needed for high recall)

Protocol is identical to E12 (same seeds, same 200/200 reference pools),
so the numbers are directly comparable to E12 ROC-AUCs.

Outputs:
  data/results/E14_operating_point_e12.csv
    dataset, embedder, seed, method, tpr_fpr1, tpr_fpr5, tpr_fpr10,
    fpr_tpr90, brier, ece10
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    score_mean_direction, score_discriminant_mean,
)
from E8_sigma_w_whitening import (  # noqa: E402
    fit_sigma_t_whitener, PooledWithinClassWhitening, score_discriminant,
)
from E13_cross_domain_transfer import (  # noqa: E402
    load_all_datasets, cache_available, get_emb,
)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "results"
N_REF = 200
N_SEEDS = 5
N_SEEDS_QWEN = 1
QWEN_MODEL = "Qwen/Qwen3-Embedding-8B"

EMBEDDERS = [
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    QWEN_MODEL,
]

def seed_count(model_id):
    return N_SEEDS_QWEN if model_id == QWEN_MODEL else N_SEEDS
def tpr_at_fpr(y, sc, fpr_target):
    order = np.argsort(-sc)
    y = y[order]
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tp = fp = 0
    best = 0.0
    for l in y:
        if l == 1:
            tp += 1
        else:
            fp += 1
        if fp / n_neg > fpr_target:
            break
        best = max(best, tp / n_pos)
    return best


def fpr_at_tpr(y, sc, tpr_target):
    order = np.argsort(-sc)
    y = y[order]
    n_pos, n_neg = int(y.sum()), len(y) - int(y.sum())
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tp = fp = 0
    best_fpr = 1.0
    for l in y:
        if l == 1:
            tp += 1
        else:
            fp += 1
        if tp / n_pos >= tpr_target:
            best_fpr = min(best_fpr, fp / n_neg)
    return best_fpr


def brier_and_ece(y, sc, n_bins=10):
    """Brier and ECE on a min-max normalized (monotone, rank-preserving) score."""
    smin, smax = sc.min(), sc.max()
    if smax - smin < 1e-12:
        return 0.0, 0.0
    p = (sc - smin) / (smax - smin)
    brier = float(np.mean((p - y) ** 2))
    edges = np.linspace(0, 1, n_bins + 1)
    ece = 0.0
    for i in range(n_bins):
        m = (p > edges[i]) & (p <= edges[i + 1])
        if m.sum() == 0:
            continue
        ece += (m.sum() / len(p)) * abs(p[m].mean() - y[m].mean())
    return brier, float(ece)


def run():
    datasets = load_all_datasets()
    rows = []
    ds_list = [d for d in datasets.items() if d[0] in ("AdvBench", "HarmBench")]
    for idx, (ds_name, (texts, labels)) in enumerate(ds_list):
        print(f"[E14] dataset {idx+1}/{len(ds_list)}: {ds_name}", flush=True)
        lab = np.array(labels)
        pos, neg = np.where(lab == 1)[0], np.where(lab == 0)[0]
        for model_id in EMBEDDERS:
            ms = model_id.split("/")[-1]
            if not cache_available(ds_name, model_id):
                continue
            print(f"  [{ds_name}] {ms}: loading embeddings...", flush=True)
            emb = get_emb(ds_name, texts, model_id)
            print(f"  [{ds_name}] {ms}: shape={emb.shape}, computing...", flush=True)
            for seed in range(seed_count(model_id)):
                rng = np.random.RandomState(seed)
                rm = rng.choice(pos, N_REF, replace=False)
                rs = rng.choice(neg, N_REF, replace=False)
                Xm, Xs = emb[rm], emb[rs]
                Xc = np.vstack([Xm, Xs])
                test_idx = np.concatenate(
                    [np.setdiff1d(pos, rm), np.setdiff1d(neg, rs)])
                y = lab[test_idx]
                Xt = emb[test_idx]

                dim = emb.shape[1]
                wh_t = fit_sigma_t_whitener(Xc, dim)
                wh_w = PooledWithinClassWhitening(dim).fit(Xm, Xs)
                Xm_t, Xs_t = wh_t.transform(Xm), wh_t.transform(Xs)
                Xm_w, Xs_w = wh_w.transform(Xm), wh_w.transform(Xs)

                scores = {
                    "A1_raw": score_mean_direction(Xt, Xm),
                    "B1_raw": score_discriminant_mean(Xt, Xm, Xs),
                    "B1b_SigmaT": score_discriminant(
                        wh_t.transform(Xt), Xm_t, Xs_t),
                    "B1w_SigmaW": score_discriminant(
                        wh_w.transform(Xt), Xm_w, Xs_w),
                }
                for meth, sc in scores.items():
                    brier, ece = brier_and_ece(y, sc)
                    rows.append({
                        "dataset": ds_name, "embedder": ms, "seed": seed,
                        "method": meth,
                        "roc_auc": roc_auc_score(y, sc),
                        "tpr_fpr1": tpr_at_fpr(y, sc, 0.01),
                        "tpr_fpr5": tpr_at_fpr(y, sc, 0.05),
                        "tpr_fpr10": tpr_at_fpr(y, sc, 0.10),
                        "fpr_tpr90": fpr_at_tpr(y, sc, 0.90),
                        "brier": brier, "ece10": ece,
                    })
                print(f"  [{ds_name}] {ms} seed={seed} done", flush=True)
        print(f"  [done] {ds_name}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "E14_operating_point_e12.csv", index=False)

    print("\n=== Operating points (mean over seeds) ===", flush=True)
    print("\n=== Operating points (mean over seeds) ===", flush=True)
    g = df.groupby(["dataset", "embedder", "method"])[
        ["roc_auc", "tpr_fpr1", "tpr_fpr5", "fpr_tpr90",
         "brier", "ece10"]].mean()
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(g.round(4).to_string(), flush=True)
    print(f"\nSaved -> {OUT / 'E14_operating_point_e12.csv'}", flush=True)
    
    # --- aggregated summary (for table in REPORT.md) ---
    sum_rows = []
    for (ds, emb), g2 in df.groupby(["dataset", "embedder"]):
        for meth in ["A1_raw", "B1_raw", "B1b_SigmaT", "B1w_SigmaW"]:
            sub = g2.loc[g2.index.get_level_values("method") == meth]
            sum_rows.append({
                "dataset": ds,
                "embedder": emb,
                "method": meth,
                "mean_roc_auc": float(sub["roc_auc"].mean()),
                "std_roc_auc": float(sub["roc_auc"].std(ddof=1)),
                "mean_tpr_fpr1": float(sub["tpr_fpr1"].mean()),
                "mean_tpr_fpr5": float(sub["tpr_fpr5"].mean()),
                "mean_tpr_fpr10": float(sub["tpr_fpr10"].mean()),
                "mean_fpr_tpr90": float(sub["fpr_tpr90"].mean()),
                "mean_brier": float(sub["brier"].mean()),
                "mean_ece10": float(sub["ece10"].mean()),
            })
    sum_df = pd.DataFrame(sum_rows)
    sum_df.to_csv(OUT / "E14_operating_point_summary.csv", index=False)
    
    print("\n=== Aggregated summary ===", flush=True)
    print(sum_df.round(4).to_string(index=False), flush=True)
    print(f"\nSaved -> {OUT / 'E14_operating_point_summary.csv'}", flush=True)
    g = df.groupby(["dataset", "embedder", "method"])[
        ["roc_auc", "tpr_fpr1", "tpr_fpr5", "fpr_tpr90",
         "brier", "ece10"]].mean()
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(g.round(4).to_string(), flush=True)
    print(f"\nSaved -> {OUT / 'E14_operating_point_e12.csv'}", flush=True)


if __name__ == "__main__":
    run()