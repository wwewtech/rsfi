"""
test_report_consistency.py
================================================================================
Consistency gate: every number in docs/RESEARCH_REPORT.md Tables 1-7 must be
recomputable from committed CSVs in data/results/.

Tolerance: table values are rounded to 4 decimals; we allow |diff| <= 5e-4
for means and stds computed from per-seed CSV values.

Known exception (documented in RESEARCH_REPORT.md): Qwen3-8B rows of
Tables 1-2 have no committed per-seed CSV (legacy session log only); they are
NOT covered here by design.
"""

import numpy as np
import pandas as pd
import pytest
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "data" / "results"
TOL = 5e-4


def load(name):
    return pd.read_csv(RESULTS / name)


def mean_std(df, mask, col="roc_auc"):
    v = df.loc[mask, col].to_numpy()
    return v.mean(), v.std(ddof=1)


def assert_cell(mean, std, exp_mean, exp_std=None):
    assert abs(mean - exp_mean) <= TOL, f"mean {mean:.4f} != {exp_mean}"
    if exp_std is not None:
        assert abs(std - exp_std) <= TOL, f"std {std:.4f} != {exp_std}"


# ---------------------------------------------------------------- Table 1 / 2
# Source: E2d_safe_aware_multidataset.csv (mpnet / bge-base / bge-large).
# Qwen3-8B rows excluded: no committed per-seed CSV (documented limitation).

E2D = None

@pytest.fixture(scope="module")
def e2d():
    global E2D
    if E2D is None:
        E2D = load("E2d_safe_aware_multidataset.csv")
    return E2D


@pytest.mark.parametrize(
    "dataset,model,method,exp_m,exp_s",
    [
        # --- Table 1 (Wild) ---
        ("Wild", "all-mpnet-base-v2", "A1_naive_cosine_raw", 0.7846, 0.0047),
        ("Wild", "all-mpnet-base-v2", "A2_rsfi_svd_raw_k20", 0.7875, 0.0074),
        ("Wild", "all-mpnet-base-v2", "B1_discriminant_mean_raw", 0.8668, 0.0060),
        ("Wild", "all-mpnet-base-v2", "C1_logreg_raw", 0.8766, 0.0071),
        ("Wild", "bge-base-en-v1.5", "A1_naive_cosine_raw", 0.7708, 0.0040),
        ("Wild", "bge-base-en-v1.5", "A2_rsfi_svd_raw_k20", 0.7854, 0.0114),
        ("Wild", "bge-base-en-v1.5", "B1_discriminant_mean_raw", 0.8618, 0.0050),
        ("Wild", "bge-base-en-v1.5", "C1_logreg_raw", 0.8728, 0.0038),
        ("Wild", "bge-large-en-v1.5", "A1_naive_cosine_raw", 0.7674, 0.0065),
        ("Wild", "bge-large-en-v1.5", "A2_rsfi_svd_raw_k20", 0.7839, 0.0110),
        ("Wild", "bge-large-en-v1.5", "B1_discriminant_mean_raw", 0.8719, 0.0050),
        ("Wild", "bge-large-en-v1.5", "C1_logreg_raw", 0.8804, 0.0028),
        # --- Table 2 (ToxicChat) ---
        ("ToxicChat", "all-mpnet-base-v2", "A1_naive_cosine_raw", 0.9158, None),
        ("ToxicChat", "all-mpnet-base-v2", "A1b_cosine_whitened", 0.9613, None),
        ("ToxicChat", "all-mpnet-base-v2", "A2_rsfi_svd_raw_k20", 0.9382, None),
        ("ToxicChat", "all-mpnet-base-v2", "A3_rsfi_svd_whitened_k20", 0.8777, None),
        ("ToxicChat", "all-mpnet-base-v2", "B1_discriminant_mean_raw", 0.9509, None),
        ("ToxicChat", "all-mpnet-base-v2", "B1b_discriminant_mean_whitened", 0.9617, None),
        ("ToxicChat", "all-mpnet-base-v2", "C1_logreg_raw", 0.9702, None),
        ("ToxicChat", "all-mpnet-base-v2", "C1b_logreg_whitened", 0.9581, None),
        ("ToxicChat", "bge-base-en-v1.5", "A1_naive_cosine_raw", 0.8798, None),
        ("ToxicChat", "bge-base-en-v1.5", "B1b_discriminant_mean_whitened", 0.9524, None),
        ("ToxicChat", "bge-large-en-v1.5", "B1b_discriminant_mean_whitened", 0.9556, None),
        # --- Table 2 (Wild) ---
        ("Wild", "all-mpnet-base-v2", "A1b_cosine_whitened", 0.8318, None),
        ("Wild", "all-mpnet-base-v2", "A3_rsfi_svd_whitened_k20", 0.8075, None),
        ("Wild", "all-mpnet-base-v2", "B1b_discriminant_mean_whitened", 0.8297, None),
        ("Wild", "all-mpnet-base-v2", "C1b_logreg_whitened", 0.8208, None),
        ("Wild", "bge-base-en-v1.5", "B1b_discriminant_mean_whitened", 0.8313, None),
        ("Wild", "bge-large-en-v1.5", "B1b_discriminant_mean_whitened", 0.8323, None),
        # --- Table 2 (XSTest) ---
        ("XSTest", "all-mpnet-base-v2", "A1_naive_cosine_raw", 0.7618, None),
        ("XSTest", "all-mpnet-base-v2", "A1b_cosine_whitened", 0.8973, None),
        ("XSTest", "all-mpnet-base-v2", "A2_rsfi_svd_raw_k20", 0.8463, None),
        ("XSTest", "all-mpnet-base-v2", "A3_rsfi_svd_whitened_k20", 0.7884, None),
        ("XSTest", "all-mpnet-base-v2", "B1_discriminant_mean_raw", 0.7851, None),
        ("XSTest", "all-mpnet-base-v2", "B1b_discriminant_mean_whitened", 0.8970, None),
        ("XSTest", "all-mpnet-base-v2", "C1_logreg_raw", 0.8542, None),
        ("XSTest", "all-mpnet-base-v2", "C1b_logreg_whitened", 0.8947, None),
        ("XSTest", "bge-base-en-v1.5", "A1b_cosine_whitened", 0.8692, None),
        ("XSTest", "bge-base-en-v1.5", "B1b_discriminant_mean_whitened", 0.8680, None),
        ("XSTest", "bge-large-en-v1.5", "A1b_cosine_whitened", 0.8883, None),
        ("XSTest", "bge-large-en-v1.5", "B1b_discriminant_mean_whitened", 0.8877, None),
    ],
)
def test_tables_1_2(e2d, dataset, model, method, exp_m, exp_s):
    m, s = mean_std(
        e2d,
        (e2d.dataset == dataset) & (e2d.model == model) & (e2d.method == method),
    )
    assert_cell(m, s, exp_m, exp_s)


# -------------------------------------------------------------------- Table 3
# DeLong claims verified from E2d_delong_tests.csv: mean diff, wins, p-value.

DELONG = None

@pytest.fixture(scope="module")
def delong():
    global DELONG
    if DELONG is None:
        DELONG = load("E2d_delong_tests.csv")
    return DELONG


@pytest.mark.parametrize(
    "dataset,model,pair,exp_diff,exp_wins",
    [
        ("Wild", "all-mpnet-base-v2", "B1_vs_A1", 0.0822, 5),
        ("Wild", "bge-base-en-v1.5", "B1_vs_A1", 0.0910, 5),
        ("Wild", "bge-large-en-v1.5", "B1_vs_A1", 0.1045, 5),
        ("Wild", "all-mpnet-base-v2", "B1b_vs_C1b", 0.0089, 5),
        ("ToxicChat", "all-mpnet-base-v2", "B1_vs_A1", 0.0351, 5),
        ("ToxicChat", "all-mpnet-base-v2", "B1b_vs_C1b", 0.0036, 5),
        ("XSTest", "all-mpnet-base-v2", "B1b_vs_A1b", -0.0003, 2),
        ("XSTest", "all-mpnet-base-v2", "A3_vs_A1", 0.0266, 5),
    ],
)
def test_table_3_delong(delong, dataset, model, pair, exp_diff, exp_wins):
    sub = delong[(delong.dataset == dataset) & (delong.model == model)
                 & (delong.pair == pair)]
    assert len(sub) == 5, f"expected 5 seeds, got {len(sub)}"
    assert abs(sub.auc_diff.mean() - exp_diff) <= TOL
    wins = int((sub.auc_diff > 0).sum())
    assert wins == exp_wins, f"wins {wins} != {exp_wins}"


def test_table_3_xstest_pvalue_not_significant(delong):
    """Report claims p=0.2819 (not significant) for XSTest B1b vs A1b."""
    sub = delong[(delong.dataset == "XSTest")
                 & (delong.model == "all-mpnet-base-v2")
                 & (delong.pair == "B1b_vs_A1b")]
    assert len(sub) == 5
    assert abs(sub.p_value.mean() - 0.2819) <= TOL


# -------------------------------------------------------------------- Table 4
# Source: E8_sigma_w.csv

E8B = None

@pytest.fixture(scope="module")
def e8b():
    global E8B
    if E8B is None:
        E8B = load("E8_sigma_w.csv")
    return E8B


@pytest.mark.parametrize(
    "dataset,model,method,exp_m,exp_s",
    [
        ("ToxicChat", "all-mpnet-base-v2", "B1_raw", 0.9509, 0.0048),
        ("ToxicChat", "all-mpnet-base-v2", "B1b_SigmaT_wh", 0.9617, 0.0020),
        ("ToxicChat", "all-mpnet-base-v2", "B1w_SigmaW_wh", 0.9680, 0.0038),
        ("ToxicChat", "bge-base-en-v1.5", "B1w_SigmaW_wh", 0.9616, 0.0045),
        ("ToxicChat", "bge-large-en-v1.5", "B1w_SigmaW_wh", 0.9637, 0.0006),
        ("Wild", "all-mpnet-base-v2", "B1_raw", 0.8668, 0.0060),
        ("Wild", "all-mpnet-base-v2", "B1b_SigmaT_wh", 0.8297, 0.0090),
        ("Wild", "all-mpnet-base-v2", "B1w_SigmaW_wh", 0.8475, 0.0093),
        ("Wild", "bge-base-en-v1.5", "B1w_SigmaW_wh", 0.8485, 0.0058),
        ("Wild", "bge-large-en-v1.5", "B1w_SigmaW_wh", 0.8479, 0.0105),
        ("XSTest", "all-mpnet-base-v2", "B1_raw", 0.7851, 0.0203),
        ("XSTest", "all-mpnet-base-v2", "B1b_SigmaT_wh", 0.8970, 0.0132),
        ("XSTest", "all-mpnet-base-v2", "B1w_SigmaW_wh", 0.8999, 0.0126),
        ("XSTest", "bge-base-en-v1.5", "B1w_SigmaW_wh", 0.8732, 0.0218),
        ("XSTest", "bge-large-en-v1.5", "B1w_SigmaW_wh", 0.8881, 0.0146),
    ],
)
def test_table_4(e8b, dataset, model, method, exp_m, exp_s):
    m, s = mean_std(
        e8b,
        (e8b.dataset == dataset) & (e8b.model == model) & (e8b.method == method),
    )
    assert_cell(m, s, exp_m, exp_s)


def test_table_4_sigma_w_never_loses(e8b):
    """Headline claim: Sigma_W >= Sigma_T in ALL 9 dataset x model configs."""
    piv = e8b.pivot_table(index=["dataset", "model"], columns="method",
                          values="roc_auc", aggfunc="mean")
    assert (piv["B1w_SigmaW_wh"] >= piv["B1b_SigmaT_wh"] - TOL).all()
    assert len(piv) == 9


def test_table_4_wild_recovery_percentages(e8b):
    """Recovery percentages printed in Table 4: 48.0 / 56.4 / 39.3."""
    piv = e8b[e8b.dataset == "Wild"].pivot_table(
        index="model", columns="method", values="roc_auc", aggfunc="mean")
    recov = (piv["B1w_SigmaW_wh"] - piv["B1b_SigmaT_wh"]) / \
            (piv["B1_raw"] - piv["B1b_SigmaT_wh"]) * 100
    assert abs(recov["all-mpnet-base-v2"] - 48.0) <= 0.1
    assert abs(recov["bge-base-en-v1.5"] - 56.4) <= 0.1
    assert abs(recov["bge-large-en-v1.5"] - 39.3) <= 0.1


# -------------------------------------------------------------------- Table 5
# Source: E8_knn.csv

E8K = None

@pytest.fixture(scope="module")
def e8k():
    global E8K
    if E8K is None:
        E8K = load("E8_knn.csv")
    return E8K


@pytest.mark.parametrize(
    "dataset,model,method,exp_m,exp_s",
    [
        ("ToxicChat", "all-mpnet-base-v2", "kNN_1class", 0.9379, 0.0064),
        ("ToxicChat", "all-mpnet-base-v2", "kNN_2class", 0.9582, 0.0049),
        ("ToxicChat", "bge-base-en-v1.5", "kNN_2class", 0.9481, 0.0081),
        ("ToxicChat", "bge-large-en-v1.5", "kNN_2class", 0.9513, 0.0084),
        ("Wild", "all-mpnet-base-v2", "kNN_1class", 0.8057, 0.0088),
        ("Wild", "all-mpnet-base-v2", "kNN_2class", 0.8781, 0.0037),
        ("Wild", "bge-base-en-v1.5", "kNN_2class", 0.8653, 0.0051),
        ("Wild", "bge-large-en-v1.5", "kNN_2class", 0.8799, 0.0061),
        ("XSTest", "all-mpnet-base-v2", "kNN_1class", 0.8311, 0.0295),
        ("XSTest", "all-mpnet-base-v2", "kNN_2class", 0.8410, 0.0398),
        ("XSTest", "bge-base-en-v1.5", "kNN_2class", 0.7955, 0.0197),
        ("XSTest", "bge-large-en-v1.5", "kNN_2class", 0.7990, 0.0293),
    ],
)
def test_table_5(e8k, dataset, model, method, exp_m, exp_s):
    m, s = mean_std(
        e8k,
        (e8k.dataset == dataset) & (e8k.model == model) & (e8k.method == method),
    )
    assert_cell(m, s, exp_m, exp_s)


# -------------------------------------------------------------------- Table 6
# Source: E3_operating_point_results.csv

E3 = None

@pytest.fixture(scope="module")
def e3():
    global E3
    if E3 is None:
        E3 = load("E3_operating_point_results.csv")
    return E3


@pytest.mark.parametrize(
    "method,col,exp_m,exp_s",
    [
        ("LogReg", "roc_auc", 0.8703, 0.0037),
        ("LogReg", "pr_auc", 0.8075, 0.0093),
        ("LogReg", "tpr_at_fpr_0100", 0.3329, 0.1321),
        ("LogReg", "tpr_at_fpr_0010", 0.2246, 0.1581),
        ("RSFI-SVD", "roc_auc", 0.7566, 0.0196),
        ("RSFI-SVD", "pr_auc", 0.7231, 0.0178),
        ("RSFI-SVD", "tpr_at_fpr_0100", 0.1680, 0.0521),
        ("RSFI-SVD", "tpr_at_fpr_0010", 0.1011, 0.0589),
        ("naive_cosine", "roc_auc", 0.7805, 0.0047),
        ("naive_cosine", "pr_auc", 0.7082, 0.0061),
        ("naive_cosine", "tpr_at_fpr_0100", 0.0883, 0.0544),
        ("naive_cosine", "tpr_at_fpr_0010", 0.0640, 0.0378),
    ],
)
def test_table_6(e3, method, col, exp_m, exp_s):
    m, s = mean_std(e3, e3.method == method, col=col)
    assert_cell(m, s, exp_m, exp_s)


# -------------------------------------------------------------------- Table 7
# Source: E9_external_baselines.csv

E9 = None

@pytest.fixture(scope="module")
def e9():
    global E9
    if E9 is None:
        E9 = load("E9_external_baselines.csv")
    return E9


@pytest.mark.parametrize(
    "dataset,method,exp_m,exp_s",
    [
        ("ToxicChat", "EXT:deberta-v3-base-prompt-injection-v2", 0.5698, 0.0100),
        ("ToxicChat", "EXT:toxic-bert", 0.7934, 0.0126),
        ("ToxicChat", "OURS:B1_raw", 0.9509, 0.0048),
        ("ToxicChat", "OURS:B1b_SigmaT_wh", 0.9617, 0.0020),
        ("Wild", "EXT:deberta-v3-base-prompt-injection-v2", 0.8405, 0.0036),
        ("Wild", "EXT:toxic-bert", 0.7236, 0.0087),
        ("Wild", "OURS:B1_raw", 0.8668, 0.0060),
        ("Wild", "OURS:B1b_SigmaT_wh", 0.8297, 0.0090),
        ("XSTest", "EXT:deberta-v3-base-prompt-injection-v2", 0.4022, 0.0123),
        ("XSTest", "EXT:toxic-bert", 0.6416, 0.0178),
        ("XSTest", "OURS:B1_raw", 0.7851, 0.0203),
        ("XSTest", "OURS:B1b_SigmaT_wh", 0.8970, 0.0132),
    ],
)
def test_table_7(e9, dataset, method, exp_m, exp_s):
    m, s = mean_std(e9, (e9.dataset == dataset) & (e9.method == method))
    assert_cell(m, s, exp_m, exp_s)


def test_e9_protocol_integrity(e9):
    """Same leakage-free budgets as E2d/E8: n_test must match per dataset."""
    expected = {"ToxicChat": 4682, "Wild": 1600, "XSTest": 301}
    for ds, n in expected.items():
        assert (e9[e9.dataset == ds].n_test == n).all()