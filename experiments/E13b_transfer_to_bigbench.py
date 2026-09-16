"""
E13b_transfer_to_bigbench.py
================================================================================
Follow-up item 2 (after NEED.md tasks 1-3): extend the E13 cross-domain transfer
matrix with the BigBench safe-pool targets introduced by E16.

Motivation (verifiable): E13 is the strictest generalization benchmark in the
repository - reference pools are drawn ONLY from the train corpus and the
discriminant is applied to a different corpus WITHOUT re-calibration
(experiments/E13_cross_domain_transfer.py, docstring lines 4-19). Its target
set is the 5 original corpora (Wild / ToxicChat / XSTest / AdvBench /
HarmBench); the BigBench safe pools added by E16
(experiments/E16_cross_domain_bigbench.py) are NOT part of it. E16 measured
AUC ~1.0 for B1/B1w on those pools, but with reference pools drawn from the
SAME dataset (in-domain split), so it cannot answer "does the discriminant
trained on AdvBench transfer to a novel linguistic domain without
recalibration?".

This script answers exactly that, reusing the committed machinery verbatim:
  - E13 helpers: load_all_datasets, get_emb, cache_available,
    balanced_eval_idx, tpr_at_fpr, N_REF=200, TEST_CAP=400;
  - E16 dataset builder: build_cross_domain_datasets (4 targets:
    {AdvBench,HarmBench} x {BB_Safe, BB_Mixed});
  - methods identical to E13: A1_raw, B1_raw, B1b_SigmaT, B1w_SigmaW, plus
    the DELONG_B1w_vs_A1 rows for parity with the committed E13 CSV.

No new method is implemented. Embeddings are read from the caches written by
E16 (`emb_cache/<dataset>_<model>.npy`); any (dataset, model) pair without a
cache is SKIPPED and reported, so the run is offline and deterministic when
the caches are present (they are, as of 2026-09-16).

Outputs (data/results/):
  E13b_transfer_to_bigbench.csv          per (train, target, model, seed, method)
  E13b_transfer_to_bigbench_summary.csv  mean/std AUC over seeds
  Console: per-target comparison table (train corpus x method).

Documented degeneracy: in the same-source cells (train corpus == the malicious
side of the target, e.g. AdvBench -> AdvBench_BB_Safe) the ranking can be
perfect for BOTH compared methods (AUC = 1.0), which makes the DeLong variance
of the difference zero, so the p-value is undefined (NaN). NaN therefore occurs
ONLY in those cells; tests/test_e13b_transfer_to_bigbench.py gates this.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score, roc_auc_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    N_SEEDS,
    delong_test,
    score_discriminant_mean,
    score_mean_direction,
)
from E8_sigma_w_whitening import (  # noqa: E402
    PooledWithinClassWhitening,
    fit_sigma_t_whitener,
    score_discriminant,
)
from E12_advbench_harmbench_extension import (  # noqa: E402
    EMBEDDERS,
    load_alpaca_selfcontained,
)
from E13_cross_domain_transfer import (  # noqa: E402
    N_REF,
    balanced_eval_idx,
    cache_available,
    get_emb,
    load_all_datasets,
    tpr_at_fpr,
)
from E16_cross_domain_bigbench import build_cross_domain_datasets  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "results"
TARGETS = ("AdvBench_BB_Safe", "AdvBench_BB_Mixed",
           "HarmBench_BB_Safe", "HarmBench_BB_Mixed")


def _auc(y, sc):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(roc_auc_score(y, sc))


def _ap(y, sc):
    if len(np.unique(y)) < 2:
        return np.nan
    return float(average_precision_score(y, sc))


def main(smoke: bool = False):
    print("=" * 80)
    print("E13b: TRANSFER INTO THE BIGBENCH SAFE-POOL DOMAINS (E13 protocol)")
    print("Refs drawn ONLY from the train corpus; targets scored WITHOUT recalibration")
    print("=" * 80, flush=True)

    train_sets = load_all_datasets()
    bb = build_cross_domain_datasets(load_alpaca_selfcontained())
    targets = {k: bb[k] for k in (TARGETS[:1] if smoke else TARGETS)}
    models = EMBEDDERS[:1] if smoke else EMBEDDERS
    n_seeds = 1 if smoke else N_SEEDS

    for name, (texts, labels) in train_sets.items():
        print(f"  [train {name}] {len(texts)} texts, "
              f"{int(np.sum(labels))} mal / "
              f"{int(len(labels) - np.sum(labels))} safe")
    for name, (texts, labels) in targets.items():
        print(f"  [target {name}] {len(texts)} texts, "
              f"{int(np.sum(labels))} mal / "
              f"{int(len(labels) - np.sum(labels))} safe")

    rows = []
    for train_ds, (tr_texts, tr_labels) in train_sets.items():
        tr_lab = np.asarray(tr_labels)
        tr_pos = np.where(tr_lab == 1)[0]
        tr_neg = np.where(tr_lab == 0)[0]

        for model_id in models:
            model_short = model_id.split("/")[-1]
            if not cache_available(train_ds, model_id):
                print(f"  [skip] no cache for train={train_ds} x {model_short}")
                continue
            tr_emb = get_emb(train_ds, tr_texts, model_id)
            print(f"\n--- train={train_ds} x {model_short} ---", flush=True)

            for seed in range(n_seeds):
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

                for tgt_ds, (te_texts, te_labels) in targets.items():
                    if not cache_available(tgt_ds, model_id):
                        print(f"    [skip] no cache for target={tgt_ds} "
                              f"x {model_short}")
                        continue
                    te_emb = get_emb(tgt_ds, te_texts, model_id)
                    te_lab = np.asarray(te_labels)
                    idx = balanced_eval_idx(te_lab, seed=seed)
                    y = te_lab[idx]
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
                            "method": meth, "n_test": int(len(y)),
                            "roc_auc": _auc(y, sc), "pr_auc": _ap(y, sc),
                            "tpr_fpr1": tpr_at_fpr(y, sc, 0.01),
                            "tpr_fpr5": tpr_at_fpr(y, sc, 0.05),
                        })
                    rows.append({
                        "train_ds": train_ds, "target_ds": tgt_ds,
                        "embedder": model_short, "seed": seed,
                        "method": "DELONG_B1w_vs_A1",
                        "roc_auc": delong_test(y, scores["B1w_SigmaW"],
                                               scores["A1_raw"]),
                        "pr_auc": np.nan, "tpr_fpr1": np.nan,
                        "tpr_fpr5": np.nan, "n_test": int(len(y)),
                    })
            print(f"  [done] {train_ds}", flush=True)

    df = pd.DataFrame(rows)
    OUT.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT / "E13b_transfer_to_bigbench.csv", index=False)

    summ = df.groupby(["train_ds", "target_ds", "embedder", "method"])[
        "roc_auc"].agg(["mean", "std"]).reset_index()
    summ.columns = ["train_ds", "target_ds", "embedder", "method",
                    "mean_auc", "std_auc"]
    summ.to_csv(OUT / "E13b_transfer_to_bigbench_summary.csv", index=False)

    print("\n=== E13b: ROC-AUC (mean over seeds and embedders), train -> BigBench ===")
    score_rows = df[df.method != "DELONG_B1w_vs_A1"]
    piv = score_rows.pivot_table(index=["target_ds", "train_ds"],
                                 columns="method", values="roc_auc")
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(piv.round(4).to_string())
    print(f"\nSaved {len(df)} rows -> {OUT / 'E13b_transfer_to_bigbench.csv'}")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="E13b: E13 transfer protocol into BigBench safe-pool targets")
    parser.add_argument("--smoke", action="store_true",
                        help="1 target set, 1 embedder, 1 seed")
    args = parser.parse_args()

    main(smoke=args.smoke)