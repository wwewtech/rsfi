"""
E17c_judge_power_analysis.py
================================================================================
Follow-up item 3 (after NEED.md tasks 1-3): statistical power of the E17
LLM-as-a-Judge measurements.

Problem (verifiable from the committed E17 outputs): the E17 summary reports
TPR = share of judge-labelled HARMFUL_COMPLIANCE generations blocked by the
filter. Most datasets produced 0 such generations, and ToxicChat produced 2
(data/results/E17_llm_judge_summary.csv, column n_harmful). A TPR estimate
backed by n_harmful <= 2 has a confidence interval so wide that no claim can
be made - this script quantifies exactly how wide, and how many harmful
generations would be required for a usable interval. All of this is closed-form
arithmetic on the committed per-prompt CSV; no GPU, no model call.

Method:
  - Wilson score interval for a binomial proportion (Wilson 1927, "Probable
    inference, the law of succession, and statistical inference", JASA 22(158),
    209-212) - used instead of the normal approximation because it stays inside
    [0, 1] and behaves at k = 0 and small n.
  - Required n for a target half-width h at observed proportion p_hat:
    n = ceil(z^2 * p_hat * (1 - p_hat) / h^2), the standard normal-approximation
    sample-size formula; reported for h = 0.05 / 0.10 / 0.20.

Inputs (data/results/):
  E17_llm_judge_prompts.csv, E17_llm_judge_summary.csv

Output (data/results/):
  E17c_judge_power_analysis.csv - per dataset x method:
    n_malicious_prompts, n_compliant (harmful generations), compliance_rate
    + Wilson CI, n_harmful_responses, tpr + Wilson CI, fpr + Wilson CI,
    required n for TPR half-widths, and the CI width actually achieved.
"""

import sys
from pathlib import Path

import numpy as np
import pandas as pd

ROOT = Path(__file__).parent.parent
RESULTS = ROOT / "data" / "results"
Z = 1.959963984540054  # two-sided 95% normal quantile


def wilson_ci(k: int, n: int, z: float = Z) -> tuple:
    """Wilson score interval for k successes out of n trials."""
    if n <= 0:
        return (np.nan, np.nan)
    p = k / n
    denom = 1.0 + z * z / n
    centre = (p + z * z / (2.0 * n)) / denom
    half = (z * np.sqrt(p * (1.0 - p) / n + z * z / (4.0 * n * n))) / denom
    return (max(0.0, centre - half), min(1.0, centre + half))


def required_n(half_width: float, p: float, z: float = Z) -> int:
    """Normal-approximation sample size for a target half-width."""
    if half_width <= 0:
        raise ValueError("half_width must be positive")
    p = min(max(float(p), 1e-9), 1 - 1e-9)
    return int(np.ceil(z * z * p * (1.0 - p) / (half_width * half_width)))
def main():
    prompts = pd.read_csv(RESULTS / "E17_llm_judge_prompts.csv")
    summary = pd.read_csv(RESULTS / "E17_llm_judge_summary.csv")

    rows = []
    for _, row in summary.iterrows():
        ds, method = row["dataset"], row["method"]
        sub = prompts[prompts.dataset == ds]
        mal_prompts = sub[sub.prompt_class == "malicious"]
        n_mal_prompts = int(len(mal_prompts))
        n_compliant = int((mal_prompts.judge_label == "HARMFUL_COMPLIANCE").sum())

        harmful = sub[sub.true_is_harmful == True]  # noqa: E712
        benign = sub[sub.true_is_harmful == False]  # noqa: E712
        n_harmful = int(len(harmful))
        n_benign = int(len(benign))

        col = "blocked_b1" if method == "B1" else "blocked_rsfi"
        k_tp = int(harmful[col].sum()) if n_harmful else 0
        k_fp = int(benign[col].sum()) if n_benign else 0

        tpr = k_tp / n_harmful if n_harmful else np.nan
        fpr = k_fp / n_benign if n_benign else np.nan
        tpr_lo, tpr_hi = wilson_ci(k_tp, n_harmful)
        fpr_lo, fpr_hi = wilson_ci(k_fp, n_benign)
        comp_lo, comp_hi = wilson_ci(n_compliant, n_mal_prompts)

        p_for_n = tpr if n_harmful else (n_compliant / n_mal_prompts
                                         if n_mal_prompts else np.nan)
        # The normal-approximation sample-size formula is not usable at a
        # boundary rate (p = 0 or 1): report NaN instead of a spurious n.
        p_usable = np.isfinite(p_for_n) and 0.0 < p_for_n < 1.0
        p_comp = n_compliant / n_mal_prompts if n_mal_prompts else np.nan
        comp_usable = np.isfinite(p_comp) and 0.0 < p_comp < 1.0
        rows.append({
            "dataset": ds, "method": method,
            "n_malicious_prompts": n_mal_prompts,
            "n_compliant": n_compliant,
            "compliance_rate": p_comp,
            "compliance_ci_low": comp_lo, "compliance_ci_high": comp_hi,
            "n_harmful_responses": n_harmful,
            "n_benign_responses": n_benign,
            "n_tp": k_tp, "n_fp": k_fp,
            "tpr": tpr, "tpr_ci_low": tpr_lo, "tpr_ci_high": tpr_hi,
            "tpr_ci_width": (tpr_hi - tpr_lo) if n_harmful else np.nan,
            "fpr": fpr, "fpr_ci_low": fpr_lo, "fpr_ci_high": fpr_hi,
            "required_n_tpr_h05": (required_n(0.05, p_for_n)
                                   if p_usable else np.nan),
            "required_n_tpr_h10": (required_n(0.10, p_for_n)
                                   if p_usable else np.nan),
            "required_n_tpr_h20": (required_n(0.20, p_for_n)
                                   if p_usable else np.nan),
            "required_n_compliance_h10": (
                required_n(0.10, p_comp) if comp_usable else np.nan),
        })

    out = pd.DataFrame(rows)
    out.to_csv(RESULTS / "E17c_judge_power_analysis.csv", index=False)

    print("=" * 80)
    print("E17c: POWER ANALYSIS OF THE E17 JUDGE GROUND TRUTH")
    print("Wilson 95% intervals; required n = z^2 p(1-p)/h^2")
    print("=" * 80)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print(out[["dataset", "method", "n_malicious_prompts", "n_compliant",
                   "compliance_rate", "compliance_ci_low", "compliance_ci_high",
                   "n_harmful_responses", "tpr", "tpr_ci_low", "tpr_ci_high",
                   "required_n_tpr_h10"]].round(4).to_string())
    print(f"\nSaved: {RESULTS / 'E17c_judge_power_analysis.csv'}")

    print("\nReference: harmful generations needed for a 95% half-width h "
          "(n = z^2 p(1-p)/h^2)")
    ref = pd.DataFrame([
        {"assumed_rate_p": p, "half_width_h": h,
         "required_n": required_n(h, p)}
        for p in (0.05, 0.10, 0.25, 0.50)
        for h in (0.05, 0.10, 0.20)
    ])
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print(ref.pivot(index="assumed_rate_p", columns="half_width_h",
                        values="required_n").to_string())
    ref.to_csv(RESULTS / "E17c_judge_power_reference.csv", index=False)
    print(f"Saved: {RESULTS / 'E17c_judge_power_reference.csv'}")


if __name__ == "__main__":
    main()