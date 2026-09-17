"""
test_report_e15_e18.py
================================================================================
Consistency gate for docs/RESEARCH_REPORT.md sections 6.7-6.10 (E15/E16/E17/E18).

Every number cited in the new sections must be recomputable from committed
CSVs in data/results/. Follows the convention of test_report_consistency.py
(Tables 1-7): tolerance 5e-4 for rounded table values, exact match for
per-seed rows quoted verbatim from CSVs.
"""
import pandas as pd
import pytest
from pathlib import Path

RESULTS = Path(__file__).parent.parent / "data" / "results"
TOL = 5e-4


def load(name):
    return pd.read_csv(RESULTS / name)


# ------------------------------------------------------------- 6.7 / E15 ----
def test_e15_target_b1w_per_seed_mpnet():
    e15 = load("E15_defense_aware_e12.csv")
    m = (e15["attack_scenario"] == "adaptive_word_greedy_target_B1w") & \
        (e15["evaluated_method"] == "B1w_SigmaW_wh") & \
        (e15["model"] == "all-mpnet-base-v2")
    sub = e15[m].sort_values(["dataset", "seed"])
    assert len(sub) == 10
    adv = sub[sub.dataset == "AdvBench"].sort_values("seed")["asr_at_1pct_fpr"].to_numpy()
    har = sub[sub.dataset == "HarmBench"].sort_values("seed")["asr_at_1pct_fpr"].to_numpy()
    assert list(adv) == pytest.approx([0.0, 0.0, 1 / 60, 1 / 60, 0.0], abs=1e-9)
    assert list(har) == pytest.approx([1 / 60, 0.0, 0.0, 0.0, 0.0], abs=1e-9)
    assert adv.mean() == pytest.approx(0.006667, abs=TOL)
    assert har.mean() == pytest.approx(0.003333, abs=TOL)
    # 9 successes of 1800 evaluations over three embedders
    mall = (e15["attack_scenario"] == "adaptive_word_greedy_target_B1w") & \
           (e15["evaluated_method"] == "B1w_SigmaW_wh")
    assert (e15[mall]["asr_at_1pct_fpr"] * 60).sum() == pytest.approx(9.0, abs=1e-6)


def test_e15_no_transfer_and_clean_zero_mpnet():
    e15 = load("E15_defense_aware_e12.csv")
    m = (e15["evaluated_method"] == "B1w_SigmaW_wh") & \
        (e15["model"] == "all-mpnet-base-v2")
    g = e15[m].groupby("attack_scenario")["asr_at_1pct_fpr"].mean()
    for sc in ["adaptive_word_greedy_target_B1", "adaptive_affix_target_B1",
               "adaptive_combined_target_B1", "clean"]:
        assert g[sc] == pytest.approx(0.0, abs=1e-9), sc


def test_e6c_contrast_numbers():
    e6c = load("E6c_defense_aware_adaptive_attack.csv")
    m = (e6c["attack_scenario"] == "adaptive_word_greedy_target_B1w") & \
        (e6c["evaluated_method"] == "B1w_SigmaW_wh") & \
        (e6c["model"] == "all-mpnet-base-v2")
    g = e6c[m].groupby("dataset")["asr_at_1pct_fpr"].mean()
    assert g["Wild"] == pytest.approx(0.903333, abs=TOL)
    assert g["ToxicChat"] == pytest.approx(0.25, abs=TOL)
    assert g["XSTest"] == pytest.approx(0.066667, abs=TOL)
    mc = (e6c["attack_scenario"] == "clean") & \
         (e6c["evaluated_method"] == "B1w_SigmaW_wh") & \
         (e6c["model"] == "all-mpnet-base-v2")
    gc = e6c[mc].groupby("dataset")["asr_at_1pct_fpr"].mean()
    assert gc["Wild"] == pytest.approx(0.47, abs=TOL)
    assert gc["ToxicChat"] == pytest.approx(0.053333, abs=TOL)
    # attack-deltas cited in 6.7
    assert g["Wild"] - gc["Wild"] == pytest.approx(0.433333, abs=TOL)
    assert g["ToxicChat"] - gc["ToxicChat"] == pytest.approx(0.196667, abs=TOL)


# ---------------------------------------------------------------- 6.8 / E16
def test_e16_five_auc_numbers_and_no_asr():
    e16 = load("E16_cross_domain_battery.csv")
    assert "asr_at_1pct_fpr" not in e16.columns
    assert set(e16.columns) == {"dataset", "seed", "n_test", "model",
                                "method", "roc_auc", "pr_auc"}
    m = (e16["dataset"] == "HarmBench_BB_Mixed") & \
        (e16["model"] == "all-mpnet-base-v2") & \
        (e16["method"] == "B1b_SigmaT_wh")
    v = e16[m].sort_values("seed")["roc_auc"].to_numpy()
    assert list(v) == pytest.approx(
        [0.9990125, 0.99825, 0.9996625, 0.9990625, 0.9996875], abs=1e-9)
    assert v.mean() == pytest.approx(0.999135, abs=TOL)
    assert v.std(ddof=1) == pytest.approx(0.000589, abs=TOL)


# ---------------------------------------------------------------- 6.9 / E17
def test_e17_underpowered_pilot():
    s = load("E17_llm_judge_summary.csv")
    h = dict(zip(s["dataset"], s["n_harmful"]))
    assert h["ToxicChat"] == 2 and h["Wild"] == 0 and h["XSTest"] == 0
    pw = load("E17c_judge_power_analysis.csv")
    row = pw[(pw.dataset == "ToxicChat") & (pw.method == "B1")].iloc[0]
    assert row["required_n_tpr_h05"] == pytest.approx(385.0)


# --------------------------------------------------------------- 6.10 / E18
def test_e18_margin_table_budget_1_2():
    su = load("E18_b1b_margin_probe_summary.csv")
    sb = su[su["is_attack_target"] & (su["budget"] == 1.2)]
    g = sb.groupby(["attack_target", "method"])[
        ["mean_margin_clean", "mean_displacement",
         "disp_over_margin", "asr_at_1pct_fpr"]].mean()
    exp = {("B1", "B1_discriminant_mean_raw"): (0.1775, 0.1375, 0.8827, 0.4333),
           ("B1b", "B1b_SigmaT_wh"): (0.0859, 0.0376, 0.4353, 0.0000),
           ("B1w", "B1w_SigmaW_wh"): (0.3921, 0.1948, 0.4917, 0.0083)}
    for key, vals in exp.items():
        got = g.loc[key].to_numpy()
        assert list(got) == pytest.approx(list(vals), abs=TOL), key


def test_e18_b1w_dose_plateau_and_b1b_zero():
    su = load("E18_b1b_margin_probe_summary.csv")
    w = su[su["is_attack_target"] & (su["attack_target"] == "B1w")]
    g = w.groupby("budget")[["disp_over_margin", "asr_at_1pct_fpr"]].mean()
    assert g.loc[0.1, "disp_over_margin"] == pytest.approx(0.3403, abs=TOL)
    assert g.loc[1.2, "disp_over_margin"] == pytest.approx(0.4917, abs=TOL)
    assert g["disp_over_margin"].max() < 0.5
    assert g.loc[1.2, "asr_at_1pct_fpr"] == pytest.approx(0.0083, abs=TOL)
    b = su[su["is_attack_target"] & (su["attack_target"] == "B1b")]
    assert (b.groupby("budget")["asr_at_1pct_fpr"].mean() == 0.0).all()
