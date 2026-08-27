"""
E9b_external_obfuscation.py
================================================================================
External published classifiers evaluated UNDER TEST-TIME OBFUSCATION on the
identical E2d/E8/E9 leakage-free splits (5 seeds), closing the honesty caveat
of report section 6.1 ("external classifiers were not run under obfuscation").

Models (zero-shot, no calibration on our data):
  - protectai/deberta-v3-base-prompt-injection-v2  (score = softmax P(INJECTION))
  - unitary/toxic-bert                             (score = max toxic-label prob)

Protocol:
  - Datasets/budgets/seed mechanics identical to E2d/E8/E9/E6b loaders.
  - Zero-shot externals need no calibration, but the same leakage-free test
    subsets are used, so rows are directly comparable to E6b (our methods)
    and E9 (externals on clean data).
  - For each obfuscation kind ALL malicious texts are transformed and scored
    once; safe texts stay clean. Scores are deterministic per text, therefore
    'clean' rows must reproduce committed E9_external_baselines.csv means
    exactly -> enforced by a built-in sanity gate within 5e-4.

Output: data/results/E9b_external_obfuscation.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat,
    load_wild,
    load_xstest,
)
from E6b_obfuscation_boundary import OBFUSCATIONS  # noqa: E402
from E9_external_baselines import ExternalClassifier  # noqa: E402

N_SEEDS = 5


def run():
    t_start = time.time()

    datasets = {
        "ToxicChat": load_toxicchat(),
        "Wild": load_wild(),
        "XSTest": load_xstest(),
    }

    externals = [
        ExternalClassifier(
            "protectai/deberta-v3-base-prompt-injection-v2",
            score_mode="softmax_class1", label_index=1,
        ),
        ExternalClassifier("unitary/toxic-bert", score_mode="max_prob"),
    ]
    externals = [e for e in externals if e.available]
    if not externals:
        print("No external classifier could be loaded - aborting.")
        sys.exit(3)

    rows = []

    for d_name, (texts, labels) in datasets.items():
        labels = np.array(labels)
        n_mal, n_safe = int(labels.sum()), len(labels) - int(labels.sum())
        print(f"\n{'#' * 80}\nDATASET: {d_name} ({len(texts)} items: "
              f"{n_mal} mal / {n_safe} safe)\n{'#' * 80}", flush=True)

        if n_mal < 250 or n_safe < 250:
            n_ref_mal = max(10, n_mal // 3)
            n_ref_safe = max(10, n_safe // 3)
        else:
            n_ref_mal = n_ref_safe = 200

        mal_idx_arr = np.where(labels == 1)[0]
        safe_idx_arr = np.where(labels == 0)[0]
        mal_texts = [texts[i] for i in mal_idx_arr]

        safe_scores = {e.model_id: e.score_all(
            [texts[i] for i in safe_idx_arr]) for e in externals}

        mal_scores = {}
        for kind in ["clean"] + list(OBFUSCATIONS.keys()):
            fn = (lambda x: x) if kind == "clean" else OBFUSCATIONS[kind]
            texts_kind = [fn(t) for t in mal_texts]
            mal_scores[kind] = {}
            for ext in externals:
                t0 = time.time()
                mal_scores[kind][ext.model_id] = ext.score_all(texts_kind)
                print(f"  scored [{d_name}/{kind}] "
                      f"{ext.model_id.split('/')[-1]}: {len(texts_kind)} texts "
                      f"in {time.time() - t0:.0f}s", flush=True)

        for seed in range(N_SEEDS):
            np.random.seed(seed)  # identical draw order as E2d/E8/E9/E6b
            ref_mal_idx = np.random.choice(mal_idx_arr, size=n_ref_mal,
                                           replace=False)
            ref_safe_idx = np.random.choice(safe_idx_arr, size=n_ref_safe,
                                            replace=False)
            test_mal_idx = np.setdiff1d(mal_idx_arr, ref_mal_idx)
            test_safe_idx = np.setdiff1d(safe_idx_arr, ref_safe_idx)

            pos_mal = np.nonzero(np.isin(mal_idx_arr, test_mal_idx))[0]
            pos_safe = np.nonzero(np.isin(safe_idx_arr, test_safe_idx))[0]
            y_test = np.array([1] * len(test_mal_idx)
                              + [0] * len(test_safe_idx), dtype=int)

            base = dict(
                dataset=d_name, seed=seed,
                n_ref_mal=n_ref_mal, n_ref_safe=n_ref_safe,
                n_test_attack=len(test_mal_idx), n_test_safe=len(test_safe_idx),
            )

            for ext in externals:
                mid = ext.model_id
                short = mid.split("/")[-1]
                for kind in ["clean"] + list(OBFUSCATIONS.keys()):
                    sc = np.concatenate([
                        mal_scores[kind][mid][pos_mal],
                        safe_scores[mid][pos_safe],
                    ])
                    rows.append(dict(
                        base, obfuscation=kind, method=f"EXT:{short}",
                        roc_auc=float(roc_auc_score(y_test, sc)),
                        pr_auc=float(average_precision_score(y_test, sc)),
                    ))

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "E9b_external_obfuscation.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} ({len(df)} rows, {time.time() - t_start:.0f}s)",
          flush=True)

    failures = sanity_check(df)
    print_summary(df)
    if failures:
        print("\nSANITY GATE FAILURES (clean vs committed E9):")
        for f in failures:
            print("  -", f)
        sys.exit(2)
    print("\nSANITY GATE PASSED: clean rows reproduce committed E9 means.")


def sanity_check(df: pd.DataFrame, tol: float = 5e-4) -> list:
    """Clean rows must reproduce committed E9_external_baselines.csv means."""
    ref_path = Path(__file__).parent.parent / "data" / "results" / \
        "E9_external_baselines.csv"
    if not ref_path.exists():
        return ["E9_external_baselines.csv missing -> cannot verify"]
    e9 = pd.read_csv(ref_path)
    failures = []
    clean = df[df.obfuscation == "clean"]
    for (ds, meth), g in e9.groupby(["dataset", "method"]):
        exp_m = g.roc_auc.mean()
        mine = clean[(clean.dataset == ds) & (clean.method == meth)]
        if not ((clean.dataset == ds) & (clean.method == meth)).any():
            continue
        got_m = mine.roc_auc.mean()
        if abs(got_m - exp_m) > tol:
            failures.append(f"{ds}/{meth}: clean mean {got_m:.4f} != "
                            f"E9 committed {exp_m:.4f}")
    return failures


def print_summary(df: pd.DataFrame):
    piv = df.pivot_table(index=["dataset", "method"],
                         columns="obfuscation", values="roc_auc")
    order = ["clean"] + list(OBFUSCATIONS.keys())
    piv = piv[[c for c in order if c in piv.columns]]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n=== ROC-AUC means over seeds ===")
        print(piv.round(4).to_string())


if __name__ == "__main__":
    run()


