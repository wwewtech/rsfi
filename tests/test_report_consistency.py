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
# Source: E2d_safe_aware_multidataset.csv & E2q_qwen_multidataset.csv.

E2D = None

@pytest.fixture(scope="module")
def e2d():
    global E2D
    if E2D is None:
        e2d_orig = load("E2d_safe_aware_multidataset.csv")
        e2q = load("E2q_qwen_multidataset.csv")
        E2D = pd.concat([e2d_orig, e2q], ignore_index=True)
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
        ("Wild", "Qwen3-Embedding-8B", "A1_naive_cosine_raw", 0.7982, 0.0102),
        ("Wild", "Qwen3-Embedding-8B", "A2_rsfi_svd_raw_k20", 0.8012, 0.0130),
        ("Wild", "Qwen3-Embedding-8B", "B1_discriminant_mean_raw", 0.8747, 0.0090),
        ("Wild", "Qwen3-Embedding-8B", "C1_logreg_raw", 0.8847, 0.0083),
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
        ("ToxicChat", "Qwen3-Embedding-8B", "A1_naive_cosine_raw", 0.8082, None),
        ("ToxicChat", "Qwen3-Embedding-8B", "A1b_cosine_whitened", 0.9677, None),
        ("ToxicChat", "Qwen3-Embedding-8B", "A2_rsfi_svd_raw_k20", 0.8651, None),
        ("ToxicChat", "Qwen3-Embedding-8B", "A3_rsfi_svd_whitened_k20", 0.8424, None),
        ("ToxicChat", "Qwen3-Embedding-8B", "B1_discriminant_mean_raw", 0.9628, None),
        ("ToxicChat", "Qwen3-Embedding-8B", "B1b_discriminant_mean_whitened", 0.9679, None),
        ("ToxicChat", "Qwen3-Embedding-8B", "C1_logreg_raw", 0.9780, None),
        ("ToxicChat", "Qwen3-Embedding-8B", "C1b_logreg_whitened", 0.9675, None),
        # --- Table 2 (Wild) ---
        ("Wild", "all-mpnet-base-v2", "A1b_cosine_whitened", 0.8318, None),
        ("Wild", "all-mpnet-base-v2", "A3_rsfi_svd_whitened_k20", 0.8075, None),
        ("Wild", "all-mpnet-base-v2", "B1b_discriminant_mean_whitened", 0.8297, None),
        ("Wild", "all-mpnet-base-v2", "C1b_logreg_whitened", 0.8208, None),
        ("Wild", "bge-base-en-v1.5", "B1b_discriminant_mean_whitened", 0.8313, None),
        ("Wild", "bge-large-en-v1.5", "B1b_discriminant_mean_whitened", 0.8323, None),
        ("Wild", "Qwen3-Embedding-8B", "A1_naive_cosine_raw", 0.7982, None),
        ("Wild", "Qwen3-Embedding-8B", "A1b_cosine_whitened", 0.8244, None),
        ("Wild", "Qwen3-Embedding-8B", "A2_rsfi_svd_raw_k20", 0.8012, None),
        ("Wild", "Qwen3-Embedding-8B", "A3_rsfi_svd_whitened_k20", 0.8033, None),
        ("Wild", "Qwen3-Embedding-8B", "B1_discriminant_mean_raw", 0.8747, None),
        ("Wild", "Qwen3-Embedding-8B", "B1b_discriminant_mean_whitened", 0.8224, None),
        ("Wild", "Qwen3-Embedding-8B", "C1_logreg_raw", 0.8847, None),
        ("Wild", "Qwen3-Embedding-8B", "C1b_logreg_whitened", 0.8200, None),
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
        ("XSTest", "Qwen3-Embedding-8B", "A1_naive_cosine_raw", 0.7824, None),
        ("XSTest", "Qwen3-Embedding-8B", "A1b_cosine_whitened", 0.9381, None),
        ("XSTest", "Qwen3-Embedding-8B", "A2_rsfi_svd_raw_k20", 0.8461, None),
        ("XSTest", "Qwen3-Embedding-8B", "A3_rsfi_svd_whitened_k20", 0.7764, None),
        ("XSTest", "Qwen3-Embedding-8B", "B1_discriminant_mean_raw", 0.8269, None),
        ("XSTest", "Qwen3-Embedding-8B", "B1b_discriminant_mean_whitened", 0.9381, None),
        ("XSTest", "Qwen3-Embedding-8B", "C1_logreg_raw", 0.9294, None),
        ("XSTest", "Qwen3-Embedding-8B", "C1b_logreg_whitened", 0.9377, None),
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
# Source: E8_sigma_w.csv & E8q_qwen_sigma_w.csv

E8B = None

@pytest.fixture(scope="module")
def e8b():
    global E8B
    if E8B is None:
        e8b_orig = load("E8_sigma_w.csv")
        e8q = load("E8q_qwen_sigma_w.csv")
        E8B = pd.concat([e8b_orig, e8q], ignore_index=True)
    return E8B


@pytest.mark.parametrize(
    "dataset,model,method,exp_m,exp_s",
    [
        ("ToxicChat", "all-mpnet-base-v2", "B1_raw", 0.9509, 0.0048),
        ("ToxicChat", "all-mpnet-base-v2", "B1b_SigmaT_wh", 0.9617, 0.0020),
        ("ToxicChat", "all-mpnet-base-v2", "B1w_SigmaW_wh", 0.9680, 0.0038),
        ("ToxicChat", "bge-base-en-v1.5", "B1w_SigmaW_wh", 0.9616, 0.0045),
        ("ToxicChat", "bge-large-en-v1.5", "B1w_SigmaW_wh", 0.9637, 0.0006),
        ("ToxicChat", "Qwen3-Embedding-8B", "B1_raw", 0.9628, 0.0031),
        ("ToxicChat", "Qwen3-Embedding-8B", "B1b_SigmaT_wh", 0.9679, 0.0045),
        ("ToxicChat", "Qwen3-Embedding-8B", "B1w_SigmaW_wh", 0.9716, 0.0035),
        ("Wild", "all-mpnet-base-v2", "B1_raw", 0.8668, 0.0060),
        ("Wild", "all-mpnet-base-v2", "B1b_SigmaT_wh", 0.8297, 0.0090),
        ("Wild", "all-mpnet-base-v2", "B1w_SigmaW_wh", 0.8475, 0.0093),
        ("Wild", "bge-base-en-v1.5", "B1w_SigmaW_wh", 0.8485, 0.0058),
        ("Wild", "bge-large-en-v1.5", "B1w_SigmaW_wh", 0.8479, 0.0105),
        ("Wild", "Qwen3-Embedding-8B", "B1_raw", 0.8747, 0.0090),
        ("Wild", "Qwen3-Embedding-8B", "B1b_SigmaT_wh", 0.8224, 0.0069),
        ("Wild", "Qwen3-Embedding-8B", "B1w_SigmaW_wh", 0.8391, 0.0049),
        ("XSTest", "all-mpnet-base-v2", "B1_raw", 0.7851, 0.0203),
        ("XSTest", "all-mpnet-base-v2", "B1b_SigmaT_wh", 0.8970, 0.0132),
        ("XSTest", "all-mpnet-base-v2", "B1w_SigmaW_wh", 0.8999, 0.0126),
        ("XSTest", "bge-base-en-v1.5", "B1w_SigmaW_wh", 0.8732, 0.0218),
        ("XSTest", "bge-large-en-v1.5", "B1w_SigmaW_wh", 0.8881, 0.0146),
        ("XSTest", "Qwen3-Embedding-8B", "B1_raw", 0.8269, 0.0316),
        ("XSTest", "Qwen3-Embedding-8B", "B1b_SigmaT_wh", 0.9381, 0.0115),
        ("XSTest", "Qwen3-Embedding-8B", "B1w_SigmaW_wh", 0.9370, 0.0113),
    ],
)
def test_table_4(e8b, dataset, model, method, exp_m, exp_s):
    m, s = mean_std(
        e8b,
        (e8b.dataset == dataset) & (e8b.model == model) & (e8b.method == method),
    )
    assert_cell(m, s, exp_m, exp_s)


def test_table_4_sigma_w_standard_models_never_lose(e8b):
    """Headline claim: Sigma_W >= Sigma_T in ALL 9 configs of standard embedders (mpnet, bge-base, bge-large)."""
    piv = e8b[e8b.model != "Qwen3-Embedding-8B"].pivot_table(
        index=["dataset", "model"], columns="method", values="roc_auc", aggfunc="mean")
    assert (piv["B1w_SigmaW_wh"] >= piv["B1b_SigmaT_wh"] - TOL).all()
    assert len(piv) == 9


def test_table_4_sigma_w_qwen_wild_toxic_win_xstest_equiv(e8b):
    """For Qwen3-8B (d=4096): Sigma_W significantly beats Sigma_T on Wild and ToxicChat; on XSTest they are equivalent."""
    piv = e8b[e8b.model == "Qwen3-Embedding-8B"].pivot_table(
        index="dataset", columns="method", values="roc_auc", aggfunc="mean")
    assert piv.loc["Wild", "B1w_SigmaW_wh"] > piv.loc["Wild", "B1b_SigmaT_wh"] + 0.015
    assert piv.loc["ToxicChat", "B1w_SigmaW_wh"] > piv.loc["ToxicChat", "B1b_SigmaT_wh"] + 0.003
    assert abs(piv.loc["XSTest", "B1w_SigmaW_wh"] - piv.loc["XSTest", "B1b_SigmaT_wh"]) <= 0.002


def test_table_4_wild_recovery_percentages(e8b):
    """Recovery percentages printed in Table 4: 48.0 / 56.4 / 39.3 / 31.9."""
    piv = e8b[e8b.dataset == "Wild"].pivot_table(
        index="model", columns="method", values="roc_auc", aggfunc="mean")
    recov = (piv["B1w_SigmaW_wh"] - piv["B1b_SigmaT_wh"]) / \
            (piv["B1_raw"] - piv["B1b_SigmaT_wh"]) * 100
    assert abs(recov["all-mpnet-base-v2"] - 48.0) <= 0.1
    assert abs(recov["bge-base-en-v1.5"] - 56.4) <= 0.1
    assert abs(recov["bge-large-en-v1.5"] - 39.3) <= 0.1
    assert abs(recov["Qwen3-Embedding-8B"] - 31.9) <= 0.1


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


# -------------------------------------------------------------------- Table 8
# Source: E6b_obfuscation_boundary.csv

E6B = None

@pytest.fixture(scope="module")
def e6b():
    global E6B
    if E6B is None:
        E6B = load("E6b_obfuscation_boundary.csv")
    return E6B

@pytest.mark.parametrize(
    "dataset,method,clean,b64,leet,rot,zw,homo",
    [
        # --- Table 8 (Wild) ---
        ("Wild", "A1_naive_cosine_raw", 0.7846, 0.2382, 0.4874, 0.5855, 0.7845, 0.5534),
        ("Wild", "A2_rsfi_svd_raw_k20", 0.7875, 0.1491, 0.3956, 0.4823, 0.7875, 0.4223),
        ("Wild", "B1_discriminant_mean_raw", 0.8668, 0.7048, 0.7792, 0.7601, 0.8668, 0.7698),
        ("Wild", "B1b_SigmaT_wh", 0.8297, 0.4511, 0.8121, 0.7906, 0.8297, 0.7189),
        ("Wild", "B1w_SigmaW_wh", 0.8475, 0.5129, 0.8331, 0.8237, 0.8475, 0.7385),
        ("Wild", "C1_logreg_raw", 0.8766, 0.7175, 0.85, 0.8164, 0.8766, 0.7881),
        # --- Table 8 (ToxicChat) ---
        ("ToxicChat", "A1_naive_cosine_raw", 0.9158, 0.7471, 0.7681, 0.8269, 0.9158, 0.8411),
        ("ToxicChat", "A2_rsfi_svd_raw_k20", 0.9382, 0.5865, 0.6222, 0.7275, 0.9382, 0.7582),
        ("ToxicChat", "B1_discriminant_mean_raw", 0.9509, 0.6621, 0.6503, 0.6964, 0.9509, 0.8351),
        ("ToxicChat", "B1b_SigmaT_wh", 0.9617, 0.6927, 0.731, 0.6353, 0.9617, 0.8044),
        ("ToxicChat", "B1w_SigmaW_wh", 0.968, 0.7242, 0.7005, 0.6828, 0.968, 0.8447),
        ("ToxicChat", "C1_logreg_raw", 0.9702, 0.711, 0.6972, 0.7141, 0.9702, 0.8486),
        # --- Table 8 (XSTest) ---
        ("XSTest", "A1_naive_cosine_raw", 0.7618, 0.1773, 0.0884, 0.0715, 0.7618, 0.1005),
        ("XSTest", "A2_rsfi_svd_raw_k20", 0.8463, 0.0487, 0.0348, 0.0096, 0.8463, 0.0808),
        ("XSTest", "B1_discriminant_mean_raw", 0.7851, 0.431, 0.4075, 0.3125, 0.7851, 0.3775),
        ("XSTest", "B1b_SigmaT_wh", 0.897, 0.2976, 0.361, 0.2845, 0.897, 0.453),
        ("XSTest", "B1w_SigmaW_wh", 0.8999, 0.311, 0.3767, 0.2901, 0.8999, 0.4485),
        ("XSTest", "C1_logreg_raw", 0.8542, 0.3125, 0.3069, 0.2069, 0.8542, 0.3269),
    ],
)
def test_table_8(e6b, dataset, method, clean, b64, leet, rot, zw, homo):
    """Every cell of report Table 8 (mpnet primary embedder)."""
    m = e6b[(e6b.dataset == dataset)
            & (e6b.model == "all-mpnet-base-v2")
            & (e6b.method == method)]
    assert len(m) == 30
    for obf, exp in [("clean", clean), ("base64", b64),
                     ("leetspeak", leet), ("rot13", rot),
                     ("zero_width", zw), ("homoglyph", homo)]:
        got = m[m.obfuscation == obf].roc_auc.mean()
        assert abs(got - exp) <= TOL, \
            f"{dataset}/{method}/{obf}: {got:.4f} != {exp}"


def test_table_8_protocol_integrity(e6b):
    """Same budgets as E2d/E9: unique n_test_attack / n_test_safe per dataset."""
    att = {"ToxicChat": 184, "Wild": 800, "XSTest": 134}
    safe = {"ToxicChat": 4498, "Wild": 800, "XSTest": 167}
    for ds, n_a in att.items():
        sub = e6b[e6b.dataset == ds]
        assert set(sub.n_test_attack.unique()) == {n_a}
        assert set(sub.n_test_safe.unique()) == {safe[ds]}


def test_e6b_clean_reproduces_committed_means(e6b):
    """Clean rows must match committed E8/E2d means within TOL"""
    specs = {
        "B1_discriminant_mean_raw": ("E8_sigma_w.csv", "B1_raw"),
        "B1b_SigmaT_wh": ("E8_sigma_w.csv", "B1b_SigmaT_wh"),
        "B1w_SigmaW_wh": ("E8_sigma_w.csv", "B1w_SigmaW_wh"),
        "A1_naive_cosine_raw": ("E2d_safe_aware_multidataset.csv",
                                "A1_naive_cosine_raw"),
        "A2_rsfi_svd_raw_k20": ("E2d_safe_aware_multidataset.csv",
                                "A2_rsfi_svd_raw_k20"),
        "C1_logreg_raw": ("E2d_safe_aware_multidataset.csv", "C1_logreg_raw"),
    }
    clean = e6b[e6b.obfuscation == "clean"]
    for method, (fname, comm_name) in specs.items():
        ref = load(fname)
        ref = ref[ref.method == comm_name]
        mine = clean[clean.method == method]
        for (ds, mdl), g in ref.groupby(["dataset", "model"]):
            if not ((mine.dataset == ds) & (mine.model == mdl)).any():
                continue
            exp_m = g.roc_auc.mean()
            got_m = mine[(mine.dataset == ds)
                         & (mine.model == mdl)].roc_auc.mean()
            assert abs(got_m - exp_m) <= TOL, \
                f"{method}/{ds}/{mdl}: {got_m:.4f} != {exp_m:.4f} ({fname})"


def test_e6b_zero_width_is_tokenizer_noop(e6b):
    """ZWSP obfuscation is indistinguishable from clean for these tokenizers:
    per-seed max |diff| <= 2e-4 across all datasets/models/methods."""
    wide = e6b.pivot_table(index=["dataset", "model", "method", "seed"],
                           columns="obfuscation", values="roc_auc")
    diff = (wide["zero_width"] - wide["clean"]).abs().max()
    assert diff <= 2e-4, f"zero_width deviates from clean by {diff}"


# -------------------------------------------------------------------- Table 9
# Source: E9b_external_obfuscation.csv

E9B = None

@pytest.fixture(scope="module")
def e9b():
    global E9B
    if E9B is None:
        E9B = load("E9b_external_obfuscation.csv")
    return E9B


@pytest.mark.parametrize(
    "dataset,method,clean,b64,leet,rot,zw,homo",
    [
        # --- Table 9 (ToxicChat) ---
        ("ToxicChat", "EXT:deberta-v3-base-prompt-injection-v2",
         0.5698, 0.9620, 0.9739, 0.8462, 0.9918, 0.9682),
        ("ToxicChat", "EXT:toxic-bert",
         0.7934, 0.8833, 0.9242, 0.8791, 0.7934, 0.9481),
        # --- Table 9 (Wild) ---
        ("Wild", "EXT:deberta-v3-base-prompt-injection-v2",
         0.8405, 0.9007, 0.9170, 0.8545, 0.9396, 0.9034),
        ("Wild", "EXT:toxic-bert",
         0.7236, 0.8844, 0.9298, 0.8581, 0.7237, 0.9624),
        # --- Table 9 (XSTest) ---
        ("XSTest", "EXT:deberta-v3-base-prompt-injection-v2",
         0.4022, 0.9969, 0.9998, 0.9742, 1.0000, 0.9962),
        ("XSTest", "EXT:toxic-bert",
         0.6416, 0.8688, 0.8497, 0.7842, 0.6416, 0.9107),
    ],
)
def test_table_9(e9b, dataset, method, clean, b64, leet, rot, zw, homo):
    """Every cell of report Table 9 (external classifiers under obfuscation)."""
    m = e9b[(e9b.dataset == dataset) & (e9b.method == method)]
    assert len(m) == 30  # 6 obfuscations x 5 seeds
    for obf, exp in [("clean", clean), ("base64", b64),
                     ("leetspeak", leet), ("rot13", rot),
                     ("zero_width", zw), ("homoglyph", homo)]:
        got = m[m.obfuscation == obf].roc_auc.mean()
        assert abs(got - exp) <= TOL, \
            f"{dataset}/{method}/{obf}: {got:.4f} != {exp}"


def test_table_9_protocol_integrity(e9b):
    """Same budgets as E2d/E8/E9/E6b: unique n_test_attack / n_test_safe per dataset."""
    att = {"ToxicChat": 184, "Wild": 800, "XSTest": 134}
    safe = {"ToxicChat": 4498, "Wild": 800, "XSTest": 167}
    for ds, n_a in att.items():
        sub = e9b[e9b.dataset == ds]
        assert set(sub.n_test_attack.unique()) == {n_a}
        assert set(sub.n_test_safe.unique()) == {safe[ds]}


def test_e9b_clean_reproduces_e9(e9b):
    """Clean rows of E9b must match committed E9_external_baselines.csv within TOL."""
    e9 = load("E9_external_baselines.csv")
    clean = e9b[e9b.obfuscation == "clean"]
    for (ds, meth), g in e9.groupby(["dataset", "method"]):
        mine = clean[(clean.dataset == ds) & (clean.method == meth)]
        if not ((clean.dataset == ds) & (clean.method == meth)).any():
            continue
        exp_m = g.roc_auc.mean()
        got_m = mine.roc_auc.mean()
        assert abs(got_m - exp_m) <= TOL, \
            f"{ds}/{meth}: clean mean {got_m:.4f} != E9 committed {exp_m:.4f}"


def test_e9b_toxic_bert_zero_width_is_noop(e9b):
    """WordPiece in toxic-bert normalizes zero-width: max |diff| <= 2e-4."""
    tb = e9b[e9b.method == "EXT:toxic-bert"]
    wide = tb.pivot_table(index=["dataset", "seed"],
                          columns="obfuscation", values="roc_auc")
    diff = (wide["zero_width"] - wide["clean"]).abs().max()
    assert diff <= 2e-4, f"toxic-bert zero_width deviates from clean by {diff}"


# ---------------------------------------------------------------- Table 10
# Source: E6c_defense_aware_adaptive_attack.csv (Defense-Aware Adaptive Adversaries)

@pytest.fixture(scope="module")
def e6c():
    return load("E6c_defense_aware_adaptive_attack.csv")


@pytest.mark.parametrize(
    "dataset,scenario,exp_perturb,exp_sim,exp_a1,exp_a2,exp_b1,exp_b1w,exp_c1,exp_deb,exp_tb",
    [
        # --- Table 10 (Wild, all-mpnet-base-v2) ---
        ("Wild", "clean", 0.000, 1.000, 0.7967, 0.7961, 0.8860, 0.8573, 0.8953, 0.8512, 0.7154),
        ("Wild", "adaptive_word_greedy_target_B1", 0.005, 0.986, 0.7908, 0.7866, 0.8829, 0.8492, 0.8922, 0.8523, 0.7162),
        ("Wild", "adaptive_affix_target_B1", 0.000, 0.728, 0.6396, 0.6420, 0.8237, 0.7512, 0.8413, 0.8909, 0.5512),
        ("Wild", "adaptive_combined_target_B1", 0.001, 0.730, 0.6354, 0.6355, 0.8225, 0.7490, 0.8401, 0.8911, 0.5509),
        ("Wild", "adaptive_word_greedy_target_B1w", 0.019, 0.943, 0.7700, 0.7600, 0.8706, 0.8011, 0.8748, 0.8555, 0.7213),
        # --- Table 10 (ToxicChat, all-mpnet-base-v2) ---
        ("ToxicChat", "clean", 0.000, 1.000, 0.9102, 0.9310, 0.9474, 0.9676, 0.9672, 0.5842, 0.8297),
        ("ToxicChat", "adaptive_word_greedy_target_B1", 0.029, 0.930, 0.8900, 0.9105, 0.9360, 0.9508, 0.9569, 0.6109, 0.8217),
        ("ToxicChat", "adaptive_affix_target_B1", 0.000, 0.701, 0.8683, 0.8802, 0.9233, 0.9150, 0.9463, 0.8306, 0.6914),
        ("ToxicChat", "adaptive_combined_target_B1", 0.010, 0.745, 0.8445, 0.8512, 0.9113, 0.8974, 0.9372, 0.8336, 0.6887),
        ("ToxicChat", "adaptive_word_greedy_target_B1w", 0.063, 0.884, 0.8790, 0.8966, 0.9234, 0.9080, 0.9407, 0.6347, 0.8203),
        # --- Table 10 (XSTest, all-mpnet-base-v2) ---
        ("XSTest", "clean", 0.000, 1.000, 0.7448, 0.8422, 0.7741, 0.9004, 0.8487, 0.3965, 0.7097),
        ("XSTest", "adaptive_word_greedy_target_B1", 0.045, 0.945, 0.7178, 0.8009, 0.7545, 0.8723, 0.8322, 0.4038, 0.6732),
        ("XSTest", "adaptive_affix_target_B1", 0.000, 0.616, 0.5259, 0.4401, 0.6482, 0.8121, 0.7484, 0.9739, 0.4393),
        ("XSTest", "adaptive_combined_target_B1", 0.000, 0.618, 0.5250, 0.4385, 0.6476, 0.8108, 0.7479, 0.9740, 0.4394),
        ("XSTest", "adaptive_word_greedy_target_B1w", 0.099, 0.878, 0.6549, 0.7218, 0.7088, 0.7549, 0.7558, 0.3901, 0.6764),
    ],
)
def test_table_10_roc_auc(e6c, dataset, scenario, exp_perturb, exp_sim,
                          exp_a1, exp_a2, exp_b1, exp_b1w, exp_c1, exp_deb, exp_tb):
    """Verify Table 10 ROC-AUC and perturbation/similarity stats for all-mpnet-base-v2."""
    sub = e6c[(e6c.dataset == dataset) & (e6c.model == "all-mpnet-base-v2") & (e6c.attack_scenario == scenario)]
    assert len(sub) == 40  # 8 methods x 5 seeds
    
    meta_sub = sub[sub.evaluated_method == "B1_discriminant_mean_raw"]
    got_perturb = meta_sub.mean_perturb_ratio.mean()
    got_sim = meta_sub.mean_semantic_sim.mean()
    assert abs(got_perturb - exp_perturb) <= 1e-3, f"perturb {got_perturb:.3f} != {exp_perturb:.3f}"
    assert abs(got_sim - exp_sim) <= 2e-3, f"sim {got_sim:.3f} != {exp_sim:.3f}"
    
    method_map = {
        "A1_naive_cosine_raw": exp_a1,
        "A2_rsfi_svd_raw_k20": exp_a2,
        "B1_discriminant_mean_raw": exp_b1,
        "B1w_SigmaW_wh": exp_b1w,
        "C1_logreg_raw": exp_c1,
        "deberta-v3-prompt-injection-v2": exp_deb,
        "toxic-bert": exp_tb,
    }
    for meth_name, exp_val in method_map.items():
        got_val = sub[sub.evaluated_method == meth_name].roc_auc.mean()
        assert abs(got_val - exp_val) <= TOL, \
            f"{dataset}/{scenario}/{meth_name}: {got_val:.4f} != {exp_val:.4f}"
