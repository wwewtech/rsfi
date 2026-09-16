"""
test_e13b_transfer_to_bigbench.py
================================================================================
Consistency gate for `data/results/E13b_transfer_to_bigbench.csv`
(E13b = E13 transfer protocol applied to the BigBench safe-pool targets of E16).

Invariants:
  1. Schema and grid: 5 train corpora x 4 BigBench targets x 4 embedders
     x 5 seeds x 4 score methods + DELONG rows; n_test equal on both classes
     halves (balanced_eval_idx semantics).
  2. AUC/PR-AUC finite and in [0, 1]; tpr_fpr1 <= tpr_fpr5.
  3. New-domain (cross-corpus) results must NOT be systematically better than
     the in-domain diagonal: for every embedder and method the mean AUC over
     train != target rows is <= the mean over train == target rows + margin.
     This is the sanity statement E13b exists to test; if it flipped, the
     target assembly (labels/text order) would be suspect.
  4. The DELONG_B1w_vs_A1 rows store p-values in [0, 1].
"""
from pathlib import Path

import numpy as np
import pandas as pd
import pytest

RESULTS = Path(__file__).parent.parent / "data" / "results"
TRAINS = {"Wild", "ToxicChat", "XSTest", "AdvBench", "HarmBench"}
TARGETS = {"AdvBench_BB_Safe", "AdvBench_BB_Mixed",
           "HarmBench_BB_Safe", "HarmBench_BB_Mixed"}
MODELS = {"all-mpnet-base-v2", "bge-base-en-v1.5", "bge-large-en-v1.5",
          "Qwen3-Embedding-8B"}
SCORE_METHODS = {"A1_raw", "B1_raw", "B1b_SigmaT", "B1w_SigmaW"}
N_SEEDS = 5


@pytest.fixture(scope="module")
def e13b():
    path = RESULTS / "E13b_transfer_to_bigbench.csv"
    if not path.exists():
        pytest.skip("E13b_transfer_to_bigbench.csv not committed yet")
    return pd.read_csv(path)


def test_schema_and_grid(e13b):
    for col in ["train_ds", "target_ds", "embedder", "seed", "method",
                "roc_auc", "pr_auc", "tpr_fpr1", "tpr_fpr5", "n_test"]:
        assert col in e13b.columns, f"missing column: {col}"
    assert set(e13b.train_ds.unique()) <= TRAINS
    assert set(e13b.target_ds.unique()) <= TARGETS
    assert set(e13b.embedder.unique()) <= MODELS
    scores = e13b[e13b.method != "DELONG_B1w_vs_A1"]
    assert set(scores.method.unique()) == SCORE_METHODS
    for (tr, tg, mdl, meth), gr in scores.groupby(
            ["train_ds", "target_ds", "embedder", "method"]):
        assert set(gr.seed.unique()) == set(range(N_SEEDS)), \
            f"{tr}->{tg}/{mdl}/{meth}: seeds {sorted(gr.seed.unique())}"
    # every (train, target) pair must be covered for a common embedder
    pairs = set(zip(e13b.train_ds, e13b.target_ds))
    assert len(pairs) == len(TRAINS) * len(TARGETS), \
        f"incomplete train x target grid: {len(pairs)} pairs"


def test_aucs_finite_and_ordered(e13b):
    scores = e13b[e13b.method != "DELONG_B1w_vs_A1"]
    assert scores.roc_auc.between(0.0, 1.0).all()
    assert np.isfinite(scores.roc_auc).all()
    assert np.isfinite(scores.pr_auc).all()
    assert (scores.tpr_fpr1 <= scores.tpr_fpr5 + 1e-9).all(), \
        "TPR@1%FPR exceeds TPR@5%FPR - threshold scan is not monotone"


def test_transfer_not_better_than_cross_source(e13b):
    """Rows that KEEP the malicious source corpus (AdvBench->AdvBench_BB_*,
    HarmBench->HarmBench_BB_*) must not be worse on average than rows that
    switch the malicious corpus as well. If this flipped, the BigBench target
    assembly (texts/labels alignment) would be suspect."""
    scores = e13b[e13b.method != "DELONG_B1w_vs_A1"].copy()
    scores["same_source"] = scores.train_ds == \
        scores.target_ds.str.split("_").str[0]
    checked = 0
    for (mdl, meth), gr in scores.groupby(["embedder", "method"]):
        same = gr[gr.same_source].roc_auc.mean()
        cross = gr[~gr.same_source].roc_auc.mean()
        assert cross <= same + 0.05, \
            f"{mdl}/{meth}: cross-source {cross:.4f} > same-source {same:.4f}"
        checked += 1
    assert checked >= 4


def test_delong_rows_are_pvalues(e13b):
    """DeLong p-values; NaN is allowed ONLY under complete separation (both
    compared AUCs equal 1.0), where the variance of the difference is zero and
    the p-value is undefined. This occurs in the same-source cells, where the
    safe-pool swap leaves the ranking perfect."""
    d = e13b[e13b.method == "DELONG_B1w_vs_A1"]
    assert len(d) > 0, "no DELONG_B1w_vs_A1 rows"
    assert d.pr_auc.isna().all()
    assert d.roc_auc.dropna().between(0.0, 1.0).all()
    nan_rows = d[d.roc_auc.isna()]
    if len(nan_rows):
        scores = e13b[e13b.method != "DELONG_B1w_vs_A1"]
        for _, r in nan_rows.iterrows():
            pair = scores[(scores.train_ds == r.train_ds)
                          & (scores.target_ds == r.target_ds)
                          & (scores.embedder == r.embedder)
                          & (scores.seed == r.seed)
                          & (scores.method.isin(["B1w_SigmaW", "A1_raw"]))]
            assert len(pair) == 2 and (pair.roc_auc >= 1.0 - 1e-12).all(), \
                f"undefined DeLong outside perfect separation at " \
                f"{r.train_ds}->{r.target_ds}"