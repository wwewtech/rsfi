"""
E18_b1b_margin_probe.py
================================================================================
Follow-up item 1 (after NEED.md tasks 1-3): why is the E15 attack success delta
for B1b EXACTLY zero?

Observed fact (data/results/E15_defense_aware_e12.csv, mean over 5 seeds and
the 4 E6c attack scenarios, ASR@1%FPR of the attack minus the clean baseline):
  A1_naive_cosine_raw      +35.72 pp (AdvBench)  +29.25 pp (HarmBench)
  B1_discriminant_mean_raw +24.36 pp             +59.14 pp
  C1_logreg_raw             +5.50 pp             +24.72 pp
  B1w_SigmaW_wh             +0.08 pp              +0.17 pp
  B1b_SigmaT_wh              0.00 pp               0.00 pp
Two competing explanations:
  (H1) SATURATION artifact - the clean ASR of B1b at the calibrated 1%-FPR
       threshold is already 0.00% (E15 clean rows), so the score margin of the
       malicious class may simply exceed the displacement the E6c attacker can
       achieve; then "0.00" is an operating-point artifact, not robustness.
  (H2) REAL INVARIANCE - the Sigma_T-whitened geometry (Corollary 1 of
       docs/PROPOSITION_1.md: ||d||_{Sigma_T^-1} = ||d||_{Sigma_W^-1} /
       sqrt(1 + pi_m pi_s ||d||^2_{Sigma_W^-1})) suppresses the very direction
       the attack moves along, so the score genuinely cannot be displaced.

This script separates H1 from H2 with two independent measurements.

PART A - geometry check (CPU, cached embeddings, no attack):
  For every (dataset, embedder, seed) compute r_W = ||d||_{Sigma_W^-1} and
  r_T = ||d||_{Sigma_T^-1} from raw (unshrunk) pools, the measured ratio
  r_T / r_W, and the Corollary-1 prediction 1/sqrt(1 + pi_m pi_s r_W^2).
  Also repeated with Ledoit-Wolf shrinkage (the actual code path) to quantify
  the shrinkage mismatch. Output: E18_b1b_geometry_check.csv

PART B - margin / displacement / dose-response probe (GPU, reuses the E6c
  attack engine verbatim via DefenseAwareAttacker):
  For each (dataset, embedder, seed, budget, attack_target):
    - attack the same malicious subsample as E6c/E15 with attack_word_greedy;
    - score the ORIGINAL and the ADVERSARIAL texts with ALL 6 E6c detectors;
    - per sample and method record: clean score, tau_1pct / tau_5pct,
      margin = clean_score - tau_1pct, displacement = clean_score - adv_score,
      evaded flags.
  H1 predicts mean displacement << mean margin for B1b (and a ratio < 1),
  H2 predicts displacement comparable to B1's while B1b stays above tau.
  Dose-response: budget in {0.1, 0.2, 0.4, 0.8} selects the number of allowed
  substitutions (1, 2, 4, 5 under the E6c cap of 5). An extra ablation level
  `cap12` raises the cap to 12 via the new `greedy_cap` kwarg of
  DefenseAwareAttacker (documented ablation; the committed E6c/E15 numbers are
  unaffected because the default cap remains 5).
  Outputs: E18_b1b_margin_probe_per_sample.csv, E18_b1b_margin_probe_summary.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import DEVICE  # noqa: E402
from E12_advbench_harmbench_extension import (  # noqa: E402
    build_behavior_datasets,
    get_embeddings_any,
)

ROOT = Path(__file__).parent.parent
OUT = ROOT / "data" / "results"
N_REF_MAL = N_REF_SAFE = 200          # E12/E15 budget rule
DEFAULT_BUDGETS = (0.1, 0.2, 0.4, 0.8)  # -> 1, 2, 4, 5 substitutions (cap 5)
DEFAULT_TARGETS = ("B1", "B1b")       # non-whitened vs Sigma_T-whitened
EMBEDDER = "sentence-transformers/all-mpnet-base-v2"
# ---------------------------------------------------------------------------
# PART A: geometry check (Corollary 1 of docs/PROPOSITION_1.md)
# ---------------------------------------------------------------------------

def _mahal_norm(M: np.ndarray, d: np.ndarray) -> float:
    """||d||_{M^{-1}} = sqrt(d^T M^{-1} d) via a linear solve (no inverse)."""
    return float(np.sqrt(max(np.dot(d, np.linalg.solve(M, d)), 0.0)))


def geometry_check(datasets: dict, embedder: str, n_seeds: int) -> pd.DataFrame:
    from sklearn.covariance import LedoitWolf

    rows = []
    for d_name, (texts, labels) in datasets.items():
        emb = get_embeddings_any(texts, d_name, embedder)
        lab = np.asarray(labels)
        pos = np.where(lab == 1)[0]
        neg = np.where(lab == 0)[0]
        for seed in range(n_seeds):
            rng = np.random.RandomState(seed)
            rm = rng.choice(pos, min(N_REF_MAL, len(pos)), replace=False)
            rs = rng.choice(neg, min(N_REF_SAFE, len(neg)), replace=False)
            Xm, Xs = emb[rm], emb[rs]
            Xc = np.vstack([Xm, Xs])
            n_m, n_s = len(rm), len(rs)
            pi_m, pi_s = n_m / (n_m + n_s), n_s / (n_m + n_s)
            d = Xm.mean(axis=0) - Xs.mean(axis=0)

            # --- raw (unshrunk) covariances: Corollary 1 is exact here, BUT the
            # empirical Sigma_W is rank-deficient when n_ref < d (few-shot
            # regime: 400 < 768), so the raw r_W is reported with its rank and
            # condition number for transparency instead of being silently used.
            Sm = np.cov(Xm, rowvar=False, bias=True)
            Ss = np.cov(Xs, rowvar=False, bias=True)
            SW = pi_m * Sm + pi_s * Ss
            ST = np.cov(Xc, rowvar=False, bias=True)
            rank_raw = int(np.linalg.matrix_rank(SW))
            cond_raw = float(np.linalg.cond(SW))
            r_W = _mahal_norm(SW, d)
            r_T = _mahal_norm(ST, d)
            ratio_meas = r_T / max(r_W, 1e-15)
            ratio_pred = 1.0 / np.sqrt(1.0 + pi_m * pi_s * r_W ** 2)
            rel_err = abs(ratio_meas - ratio_pred) / max(ratio_pred, 1e-15)

            # --- Ledoit-Wolf shrunk covariances (the actual code path)
            lw_m, lw_s, lw_t = LedoitWolf().fit(Xm), LedoitWolf().fit(Xs), \
                LedoitWolf().fit(Xc)
            Sm_lw, Ss_lw, ST_lw = lw_m.covariance_, lw_s.covariance_, \
                lw_t.covariance_
            SW_lw = pi_m * Sm_lw + pi_s * Ss_lw
            r_W_lw = _mahal_norm(SW_lw, d)
            r_T_lw = _mahal_norm(ST_lw, d)
            ratio_meas_lw = r_T_lw / max(r_W_lw, 1e-15)
            ratio_pred_lw = 1.0 / np.sqrt(1.0 + pi_m * pi_s * r_W_lw ** 2)
            rel_err_lw = (abs(ratio_meas_lw - ratio_pred_lw)
                          / max(ratio_pred_lw, 1e-15))

            rows.append({
                "dataset": d_name, "model": embedder.split("/")[-1],
                "seed": seed, "n_ref_mal": n_m, "n_ref_safe": n_s,
                "dim": int(Xm.shape[1]), "rank_SW_raw": rank_raw,
                "cond_SW_raw": cond_raw,
                "pi_m": pi_m, "pi_s": pi_s,
                "r_W_raw": r_W, "r_T_raw": r_T,
                "ratio_meas_raw": ratio_meas, "ratio_pred_raw": ratio_pred,
                "rel_err_raw": rel_err,
                "suppression_raw": 1.0 - ratio_pred,
                "rho_m": float(lw_m.shrinkage_), "rho_s": float(lw_s.shrinkage_),
                "rho_T": float(lw_t.shrinkage_),
                "r_W_lw": r_W_lw, "r_T_lw": r_T_lw,
                "ratio_meas_lw": ratio_meas_lw,
                "ratio_pred_lw": ratio_pred_lw, "rel_err_lw": rel_err_lw,
                "suppression_lw": 1.0 - ratio_pred_lw,
            })
        sub = [r for r in rows if r["dataset"] == d_name]
        print(f"  [geometry] {d_name}: {n_seeds} seeds done, "
              f"r_W_raw mean {np.mean([r['r_W_raw'] for r in sub]):.3f}", flush=True)
    return pd.DataFrame(rows)


def _run_one_attack(model, budget, target, eval_texts, eval_mal, d_name,
                    embedder, seed, clean_scores, thresholds, Xm, Xs, wh_w,
                    wh_t, lr, w_b1, w_b1b, w_b1w, greedy_caps):
    """One (budget, attack target) cell: attack, score with all 6 detectors."""
    from E6c_defense_aware_adaptive_attack import (
        DefenseAwareAttacker,
        compute_all_detector_scores,
    )

    w, wh, tau, target_method = _attacker_spec(
        target, w_b1, w_b1b, w_b1w, wh_t, wh_w, thresholds)
    attacker = DefenseAwareAttacker(
        model=model, w_disc=w, whitener=wh, tau_threshold=tau,
        max_perturb_ratio=budget, greedy_cap=greedy_caps.get(budget, 5))
    t0 = time.time()
    res = [attacker.attack_word_greedy(p) for p in eval_texts]
    adv_texts = [r["adv_text"] for r in res]
    adv_emb = model.encode(adv_texts, convert_to_numpy=True,
                           show_progress_bar=False, batch_size=128)
    adv_scores = compute_all_detector_scores(adv_emb, Xm, Xs, wh_w, wh_t, lr)
    mean_perturb = float(np.mean([r["perturb_ratio"] for r in res]))
    mean_sim = float(np.mean([r["semantic_sim"] for r in res]))
    secs = time.time() - t0

    per_sample, summary = [], []
    for m_name, sc_adv in adv_scores.items():
        sc_clean = clean_scores[m_name]
        tau1 = thresholds[m_name]["tau_1pct"]
        tau5 = thresholds[m_name]["tau_5pct"]
        margin = sc_clean - tau1
        disp = sc_clean - sc_adv
        evaded1 = sc_adv < tau1
        evaded5 = sc_adv < tau5
        for j in range(len(eval_mal)):
            per_sample.append({
                "dataset": d_name, "model": embedder.split("/")[-1],
                "seed": seed, "budget": budget, "attack_target": target,
                "prompt_idx": int(eval_mal[j]), "method": m_name,
                "clean_score": float(sc_clean[j]),
                "tau_1pct": float(tau1), "tau_5pct": float(tau5),
                "margin_clean": float(margin[j]),
                "adv_score": float(sc_adv[j]),
                "displacement": float(disp[j]),
                "evaded_1pct": bool(evaded1[j]),
                "evaded_5pct": bool(evaded5[j]),
                "semantic_sim": float(res[j]["semantic_sim"]),
                "perturb_ratio": float(res[j]["perturb_ratio"]),
                "n_changes": int(res[j]["n_changes"]),
            })
        summary.append({
            "dataset": d_name, "model": embedder.split("/")[-1],
            "seed": seed, "budget": budget,
            "greedy_cap": greedy_caps.get(budget, 5),
            "attack_target": target, "method": m_name,
            "is_attack_target": m_name == target_method,
            "n_attack_samples": int(len(eval_mal)),
            "mean_margin_clean": float(np.mean(margin)),
            "median_margin_clean": float(np.median(margin)),
            "min_margin_clean": float(np.min(margin)),
            "mean_displacement": float(np.mean(disp)),
            "median_displacement": float(np.median(disp)),
            "disp_over_margin": float(np.mean(disp) / max(np.mean(margin), 1e-9)),
            "max_displacement": float(np.max(disp)),
            "asr_at_1pct_fpr": float(np.mean(evaded1)),
            "asr_at_5pct_fpr": float(np.mean(evaded5)),
            "mean_perturb_ratio": mean_perturb,
            "mean_semantic_sim": mean_sim,
            "mean_n_changes": float(np.mean([r["n_changes"] for r in res])),
            "seconds": secs,
        })

    key = ["B1_discriminant_mean_raw", "B1b_SigmaT_wh", "B1w_SigmaW_wh"]
    asr_txt = " ".join(
        f"{k.split('_')[0]}={np.mean(adv_scores[k] < thresholds[k]['tau_1pct']):.3f}"
        for k in key)
    print(f"  [{d_name}/s{seed}/b{budget}/t{target}] {asr_txt} "
          f"sim={mean_sim:.3f} ({secs:.0f}s)", flush=True)
    return per_sample, summary


# ---------------------------------------------------------------------------
# PART B: margin / displacement / dose-response probe (GPU, E6c engine)
# ---------------------------------------------------------------------------

def _attacker_spec(target: str, w_b1, w_b1b, w_b1w, wh_t, wh_w, thresholds):
    """(weight vector, whitener, tau, method name) for an attack target."""
    if target == "B1":
        return (w_b1, None, thresholds["B1_discriminant_mean_raw"]["tau_1pct"],
                "B1_discriminant_mean_raw")
    if target == "B1b":
        return (w_b1b, wh_t, thresholds["B1b_SigmaT_wh"]["tau_1pct"],
                "B1b_SigmaT_wh")
    if target == "B1w":
        return (w_b1w, wh_w, thresholds["B1w_SigmaW_wh"]["tau_1pct"],
                "B1w_SigmaW_wh")
    raise ValueError(f"unknown attack target: {target}")


def _seed_setup(texts, labels, seed, max_attacks, emb):
    """Replicate the E6c/E15 split rule for one (dataset, seed)."""
    lab = np.asarray(labels)
    np.random.seed(seed)
    mal_idx = np.where(lab == 1)[0]
    safe_idx = np.where(lab == 0)[0]
    ref_mal = np.random.choice(mal_idx, N_REF_MAL, replace=False)
    ref_safe = np.random.choice(safe_idx, N_REF_SAFE, replace=False)
    test_mal = np.setdiff1d(mal_idx, ref_mal)
    np.random.seed(seed + 100)
    eval_mal = (np.random.choice(test_mal, max_attacks, replace=False)
                if len(test_mal) > max_attacks else test_mal)
    return ref_mal, ref_safe, eval_mal


def margin_probe(datasets: dict, embedder: str, budgets, targets,
                 n_seeds: int, max_attacks: int, greedy_caps=None):
    from sentence_transformers import SentenceTransformer
    from sklearn.linear_model import LogisticRegression

    from E6c_defense_aware_adaptive_attack import (
        DefenseAwareAttacker,
        compute_all_detector_scores,
        compute_calibration_thresholds,
    )
    from E8_sigma_w_whitening import PooledWithinClassWhitening, \
        fit_sigma_t_whitener

    if greedy_caps is None:
        greedy_caps = {b: 5 for b in budgets}

    model = SentenceTransformer(embedder, device=DEVICE)
    per_sample, summary = [], []
    t_start = time.time()

    for d_name, (texts, labels) in datasets.items():
        emb = get_embeddings_any(texts, d_name, embedder)
        print(f"\n### margin probe: {d_name} ({len(texts)} texts)", flush=True)

        for seed in range(n_seeds):
            ref_mal, ref_safe, eval_mal = _seed_setup(
                texts, labels, seed, max_attacks, emb)
            Xm, Xs = emb[ref_mal], emb[ref_safe]
            Xc = np.vstack([Xm, Xs])
            dim = Xm.shape[1]
            wh_t = fit_sigma_t_whitener(Xc, dim)
            wh_w = PooledWithinClassWhitening(dim).fit(Xm, Xs)
            y_tr = np.concatenate([np.ones(len(Xm)), np.zeros(len(Xs))])
            lr = LogisticRegression(max_iter=1000, random_state=seed).fit(Xc, y_tr)

            thresholds = compute_calibration_thresholds(Xm, Xs, wh_w, wh_t, lr)
            w_b1 = Xm.mean(0) - Xs.mean(0)
            w_b1b = wh_t.transform(Xm).mean(0) - wh_t.transform(Xs).mean(0)
            w_b1w = wh_w.transform(Xm).mean(0) - wh_w.transform(Xs).mean(0)

            # clean scores of the attacked subsample, computed ONCE per seed
            clean_scores = compute_all_detector_scores(
                emb[eval_mal], Xm, Xs, wh_w, wh_t, lr)
            eval_texts = [texts[i] for i in eval_mal]
            print(f"  [seed {seed}] {len(eval_mal)} attack samples", flush=True)

            for budget in budgets:
                for target in targets:
                    _ps, _sum = _run_one_attack(
                        model, budget, target, eval_texts, eval_mal,
                        d_name=d_name, embedder=embedder, seed=seed,
                        clean_scores=clean_scores, thresholds=thresholds,
                        Xm=Xm, Xs=Xs, wh_w=wh_w, wh_t=wh_t, lr=lr,
                        w_b1=w_b1, w_b1b=w_b1b, w_b1w=w_b1w,
                        greedy_caps=greedy_caps)
                    per_sample.extend(_ps)
                    summary.extend(_sum)

    print(f"\n[probe] total {time.time() - t_start:.0f}s", flush=True)
    return pd.DataFrame(per_sample), pd.DataFrame(summary)


def _print_disp_vs_margin(summary: pd.DataFrame):
    """Key H1-vs-H2 table: displacement vs margin per method (attack-target rows)."""
    if summary.empty:
        return
    sub = summary[summary.is_attack_target]
    piv = sub.pivot_table(index=["dataset", "attack_target", "budget"],
                          columns="method",
                          values=["disp_over_margin", "asr_at_1pct_fpr"])
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\n=== displacement/margin and ASR@1%FPR by attack target and budget ===")
        print(piv.round(3).to_string())


def main(smoke: bool = False, geometry_only: bool = False, cap12: bool = False,
         targets=None):
    print("=" * 80)
    print("E18: WHY IS THE B1b ATTACK DELTA EXACTLY ZERO? (margin vs invariance)")
    print("=" * 80, flush=True)

    datasets = build_behavior_datasets()
    if smoke:
        datasets = {"AdvBench": datasets["AdvBench"]}

    budgets = (0.1, 0.4) if smoke else DEFAULT_BUDGETS
    targets = tuple(targets) if targets else DEFAULT_TARGETS
    n_seeds = 1 if smoke else 3
    max_attacks = 15 if smoke else 60
    print(f"targets={targets} budgets={budgets} seeds={n_seeds} "
          f"attacks/seed={max_attacks}", flush=True)

    geom = geometry_check(datasets, EMBEDDER, n_seeds)
    geom.to_csv(OUT / "E18_b1b_geometry_check.csv", index=False)
    with pd.option_context("display.max_columns", None, "display.width", 240):
        print("\n=== PART A: Corollary-1 geometry check ===")
        print("raw  = unshrunk empirical covariances (rank-deficient: n_ref < d)")
        print("lw   = Ledoit-Wolf shrunk covariances (the actual code path)")
        cols = ["dim", "rank_SW_raw", "cond_SW_raw", "rho_m", "rho_s", "rho_T",
                "r_W_lw", "r_T_lw", "ratio_meas_lw", "ratio_pred_lw",
                "rel_err_lw", "suppression_lw"]
        print(geom.groupby("dataset")[cols].mean().round(5).to_string())
    print(f"Saved: {OUT / 'E18_b1b_geometry_check.csv'}")

    if geometry_only:
        return

    if cap12:
        budgets = tuple(budgets) + (1.2,)
    greedy_caps = {b: (12 if b == 1.2 else 5) for b in budgets}

    per_sample, summary = margin_probe(
        datasets, EMBEDDER, budgets, targets, n_seeds, max_attacks,
        greedy_caps=greedy_caps)
    OUT.mkdir(parents=True, exist_ok=True)
    per_sample.to_csv(OUT / "E18_b1b_margin_probe_per_sample.csv", index=False)
    summary.to_csv(OUT / "E18_b1b_margin_probe_summary.csv", index=False)

    _print_disp_vs_margin(summary)
    print(f"\nSaved: {OUT / 'E18_b1b_margin_probe_per_sample.csv'} "
          f"({len(per_sample)} rows)")
    print(f"Saved: {OUT / 'E18_b1b_margin_probe_summary.csv'} "
          f"({len(summary)} rows)")


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(
        description="E18: B1b zero-delta probe (saturation artifact vs invariance)")
    parser.add_argument("--smoke", action="store_true",
                        help="1 dataset, 1 seed, 2 budgets, 15 attacks")
    parser.add_argument("--geometry-only", action="store_true",
                        help="run only PART A (no GPU attack)")
    parser.add_argument("--cap12", action="store_true",
                        help="add an extended-budget ablation level (cap 12)")
    parser.add_argument("--targets", type=str, default=None,
                        help="comma-separated attack targets, e.g. B1,B1b,B1w")
    args = parser.parse_args()

    tgt = args.targets.split(",") if args.targets else None
    main(smoke=args.smoke, geometry_only=args.geometry_only, cap12=args.cap12,
         targets=tgt)
