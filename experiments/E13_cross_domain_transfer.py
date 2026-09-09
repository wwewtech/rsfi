"""
E13_cross_domain_transfer.py
================================================================================
MAXIMALLY OBJECTIVE generalization benchmark: train on one dataset, score on
ALL others (including held-out domain). Unlike E2d/E8/E12 (same-domain split),
this measures how well the Safe-Aware discriminant transfers across
distributions without re-calibration.

Design (leakage-free, deterministic):
  - Reference pools (200/200) are ALWAYS drawn from the TRAIN dataset only.
  - Test scoring uses the FULL evaluation set of the TRANSFER dataset
    (balanced cap 400 for very large pools); for the in-domain case
    (target == train) the reference pool is EXCLUDED from test (leak-free).
  - 5 fixed seeds -> 5 independent reference pools per (train_ds, embedder).
  - Per seed, the SAME reference pool scores all target datasets, so
    per-seed paired comparisons are valid.
  - Methods: A1 (raw cosine), B1 (discriminant raw), B1b (discriminant after
    Sigma_T whitening on train pool), B1w (after Sigma_W pooling).
  - Qwen3-8B included only where cache files exist.

Outputs:
  data/results/E13_cross_domain_transfer.csv
    (train_ds, target_ds, embedder, seed, method, roc_auc, pr_auc,
     tpr_fpr1, tpr_fpr5, n_test)  +  DELONG_B1w_vs_A1 rows
  data/results/E13_cross_domain_summary.csv
    per (train_ds, target_ds, embedder, method): mean/std AUC over 5 seeds.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat, load_wild, load_xstest,
    score_mean_direction, score_discriminant_mean, delong_test, N_SEEDS,
)
from E8_sigma_w_whitening import (  # noqa: E402
    fit_sigma_t_whitener, PooledWithinClassWhitening, score_discriminant,
)
from E12_advbench_harmbench_extension import (  # noqa: E402
    build_behavior_datasets, get_embeddings_any,
)

ROOT = Path(__file__).parent.parent
EMB_CACHE = ROOT / "emb_cache"
OUT = ROOT / "data" / "results"

N_REF = 200
TEST_CAP = 400

EMBEDDERS = [
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "Qwen/Qwen3-Embedding-8B",
]

def load_all_datasets():
    """Unify every dataset in the repo into {name: (texts, labels)}."""
    datasets = {}
    datasets["Wild"] = load_wild()
    datasets["ToxicChat"] = load_toxicchat()
    datasets["XSTest"] = load_xstest()
    datasets["AdvBench"] = tuple(list(v) for v in build_behavior_datasets()["AdvBench"])
    datasets["HarmBench"] = tuple(list(v) for v in build_behavior_datasets()["HarmBench"])
    # Wild cache is only the 2000-row shared file; keep only that prefix.
    texts, labels = datasets["Wild"]
    datasets["Wild"] = (texts[:2000], labels[:2000])
    return datasets


def cache_available(ds_name, model_id) -> bool:
    """Whether the embedding cache the pipeline would use exists."""
    if "Qwen3" in model_id:
        fn = f"{ds_name}_Qwen_Qwen3-Embedding-8B.npy"
        if ds_name == "Wild":
            fn = "Qwen_Qwen3-Embedding-8B.npy"
        return (EMB_CACHE / fn).exists()
    safe = model_id.replace("/", "_")
    if ds_name == "Wild":
        if "bge-base" in model_id:
            return (EMB_CACHE / "BAAI_bge-base-en-v1.5.npy").exists()
        if "bge-large" in model_id:
            return (EMB_CACHE / "BAAI_bge-large-en-v1.5.npy").exists()
        if "mpnet" in model_id:
            return (EMB_CACHE / "all-mpnet-base-v2.npy").exists()
    return (EMB_CACHE / f"{ds_name}_{safe}.npy").exists()


def get_emb(ds_name, texts, model_id):
    """Cached embeddings with correct Wild fallback semantics."""
    if ds_name == "Wild" and "Qwen3" not in model_id:
        if "bge-base" in model_id:
            return np.load(EMB_CACHE / "BAAI_bge-base-en-v1.5.npy")
        if "bge-large" in model_id:
            return np.load(EMB_CACHE / "BAAI_bge-large-en-v1.5.npy")
        if "mpnet" in model_id:
            return np.load(EMB_CACHE / "all-mpnet-base-v2.npy")
    return get_embeddings_any(texts, ds_name, model_id)


def balanced_eval_idx(labels, cap=TEST_CAP, seed=0):
    """Balanced capped evaluation index for the target dataset."""
    lab = np.array(labels)
    pos = np.where(lab == 1)[0]
    neg = np.where(lab == 0)[0]
    rng = np.random.RandomState(seed)
    n_pos = min(len(pos), cap // 2)
    n_neg = min(len(neg), cap // 2)
    pi = rng.choice(pos, n_pos, replace=False)
    ni = rng.choice(neg, n_neg, replace=False)
    return np.concatenate([pi, ni])


def tpr_at_fpr(y_true, scores, fpr_target):
    """TPR at a target FPR via monotone threshold scan."""
    order = np.argsort(-scores)
    y = y_true[order]
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return np.nan
    tp, fp, best = 0, 0, 0.0
    for l in y:
        if l == 1:
            tp += 1
        else:
            fp += 1
        if fp / max(n_neg, 1) > fpr_target:
            break
        best = max(best, tp / n_pos)
    return best

def main():
    print("=" * 80)
    print("E13: CROSS-DOMAIN TRANSFER BENCHMARK (train->all target datasets)")
    print("=" * 80)

    datasets = load_all_datasets()
    for name, (texts, labels) in datasets.items():
        print(f"  [{name}] {len(texts)} texts, "
              f"{sum(labels)} mal / {len(labels) - sum(labels)} safe")

    rows = []
    for train_ds, (tr_texts, tr_labels) in datasets.items():
        tr_lab = np.array(tr_labels)
        tr_pos = np.where(tr_lab == 1)[0]
        tr_neg = np.where(tr_lab == 0)[0]

        for model_id in EMBEDDERS:
            model_short = model_id.split("/")[-1]
            if not cache_available(train_ds, model_id):
                print(f"  [skip] no cache for {train_ds} x {model_short}")
                continue
            tr_emb = get_emb(train_ds, tr_texts, model_id)
            print(f"\n--- train={train_ds} x {model_short} ---", flush=True)

            for seed in range(N_SEEDS):
                rng = np.random.RandomState(seed)
                rm = rng.choice(tr_pos, N_REF, replace=False)
                rs = rng.choice(tr_neg, N_REF, replace=False)
                Xm, Xs = tr_emb[rm], tr_emb[rs]
                Xc = np.vstack([Xm, Xs])
                dim = tr_emb.shape[1]

                wh_t = fit_sigma_t_whitener(Xc, dim)
                wh_w = PooledWithinClassWhitening(dim).fit(Xm, Xs)
                Xm_t, Xs_t = wh_t.transform(Xm), wh_t.transform(Xs)
                Xm_w, Xs_w = wh_w.transform(Xm), wh_w.transform(Xs)

                for tgt_ds, (te_texts, te_labels) in datasets.items():
                    if not cache_available(tgt_ds, model_id):
                        continue
                    te_emb = get_emb(tgt_ds, te_texts, model_id)
                    te_lab = np.array(te_labels)
                    if tgt_ds == train_ds:
                        # In-domain: EXCLUDE the reference pool from the test
                        # set (leak-free, same semantics as E12/E14 setdiff).
                        # Audit 2026-09: balanced_eval_idx(seed) selects the
                        # SAME indices as the ref-pool draw -> 100% overlap,
                        # inflating in-domain AUC to ~0.999.
                        te_pos = np.setdiff1d(np.where(te_lab == 1)[0], rm)
                        te_neg = np.setdiff1d(np.where(te_lab == 0)[0], rs)
                        # Degenerate cell (e.g. XSTest: all 200 positives fit
                        # into the ref pool -> empty positive test class).
                        if len(te_pos) == 0 or len(te_neg) == 0:
                            print(f"    [skip] {tgt_ds} diag: no held-out "
                                  f"class members left after ref exclusion")
                            continue
                        rng2 = np.random.RandomState(seed)
                        n_pos = min(len(te_pos), TEST_CAP // 2)
                        n_neg = min(len(te_neg), TEST_CAP // 2)
                        pi = rng2.choice(te_pos, n_pos, replace=False)
                        ni = rng2.choice(te_neg, n_neg, replace=False)
                        idx = np.concatenate([pi, ni])
                    else:
                        idx = balanced_eval_idx(te_lab, seed=seed)
                    y = np.array(te_labels)[idx]
                    Xt = te_emb[idx]

                    scores = {
                        "A1_raw": score_mean_direction(Xt, Xm),
                        "B1_raw": score_discriminant_mean(Xt, Xm, Xs),
                        "B1b_SigmaT": score_discriminant(
                            wh_t.transform(Xt), Xm_t, Xs_t),
                        "B1w_SigmaW": score_discriminant(
                            wh_w.transform(Xt), Xm_w, Xs_w),
                    }
                    for meth, sc in scores.items():
                        rows.append({
                            "train_ds": train_ds, "target_ds": tgt_ds,
                            "embedder": model_short, "seed": seed,
                            "method": meth, "n_test": len(y),
                            "roc_auc": roc_auc_score(y, sc),
                            "pr_auc": average_precision_score(y, sc),
                            "tpr_fpr1": tpr_at_fpr(y, sc, 0.01),
                            "tpr_fpr5": tpr_at_fpr(y, sc, 0.05),
                        })
                    rows.append({
                        "train_ds": train_ds, "target_ds": tgt_ds,
                        "embedder": model_short, "seed": seed,
                        "method": "DELONG_B1w_vs_A1",
                        "roc_auc": delong_test(y, scores["B1w_SigmaW"],
                                               scores["A1_raw"]),
                        "pr_auc": np.nan,
                        "tpr_fpr1": np.nan, "tpr_fpr5": np.nan,
                        "n_test": len(y),
                    })
        print(f"  [done] {train_ds}")

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "E13_cross_domain_transfer.csv", index=False)

    summ = df.groupby(["train_ds", "target_ds", "embedder", "method"])[
        "roc_auc"].agg(["mean", "std"]).reset_index()
    summ.columns = ["train_ds", "target_ds", "embedder", "method",
                    "mean_auc", "std_auc"]
    summ.to_csv(OUT / "E13_cross_domain_summary.csv", index=False)

    print("\n=== Transfer AUC (B1w, mean over seeds), train->target ===")
    piv = df[df.method == "B1w_SigmaW"].pivot_table(
        index=["train_ds", "target_ds"], columns="embedder", values="roc_auc")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(piv.round(4).to_string())
    print(f"\nSaved {len(df)} rows -> {OUT / 'E13_cross_domain_transfer.csv'}")


if __name__ == "__main__":
    main()