"""
E6b_obfuscation_boundary.py
================================================================================
Boundary of applicability under OBFUSCATION for the CENTRAL paper methods
(B1 / B1b / B1w), closing audit gap "E6 tied only to legacy RSFI-SVD".

Motivation (external review, 26.08.2026):
    The previous E6 artifact was degenerate: its committed CSV has n_ref=5,
    n_test_attack=5 (ten attack strings total), a single embedder, and scored
    only the one-class methods (RSFI-SVD, naive cosine). The central Safe-Aware
    methods of the paper were never tested against obfuscation.

Protocol (identical splits/budgets/seeds as E2d/E8/E9):
    - Datasets & loaders, budget rule (//3 or 200/200), np.random.seed(seed),
      setdiff1d leakage-free splits are reused verbatim from
      E2d_safe_aware_multidataset.py.
    - Calibration uses CLEAN malicious references ONLY (deployment-realistic:
      the whitener/discriminant is fitted before the attacker adapts);
      test attacks are the obfuscated remainder; safe texts stay clean.
    - Obfuscations: base64, leetspeak, rot13, zero_width, homoglyph
      (+ 'clean' reference rows which MUST reproduce committed E8/E2d means,
      enforced in-script within 5e-4 -> guard against silent protocol drift).
    - Embedders: mpnet / bge-base / bge-large (same 3 as Tables 1-7);
      obfuscated embeddings are cached in emb_cache/{ds}__obf-{kind}_{model}.npy.
    - Metrics: ROC-AUC, PR-AUC per (dataset, model, seed, obfuscation, method).

Methods:
    A1_naive_cosine_raw        - blind anchor (legacy baseline continuity)
    A2_rsfi_svd_raw_k20        - legacy one-class RSFI-SVD anchor
    B1_discriminant_mean_raw   - central method, raw space
    B1b_SigmaT_wh              - central method, ZCA(Sigma_T) whitened (E8 impl)
    B1w_SigmaW_wh              - recommended variant, pooled Sigma_W whitening
    C1_logreg_raw              - supervised ceiling reference

Output: data/results/E6b_obfuscation_boundary.csv
"""

import sys
import time
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat,
    load_wild,
    load_xstest,
    get_embeddings,
    score_mean_direction,
    score_svd_subspace,
    DEVICE,
)
from E8_sigma_w_whitening import (  # noqa: E402
    fit_sigma_t_whitener,
    PooledWithinClassWhitening,
    score_discriminant,
)

N_SEEDS = 5

# ---------------------------------------------------------------------------
# Obfuscations (verbatim from legacy experiments/E6_adaptive_attacks.py)
# ---------------------------------------------------------------------------

def obfuscate_base64(text: str) -> str:
    return base64.b64encode(text.encode()).decode()


def obfuscate_leetspeak(text: str) -> str:
    leet_map = {
        "a": "4", "A": "4", "e": "3", "E": "3", "i": "1", "I": "1",
        "o": "0", "O": "0", "s": "5", "S": "5", "t": "7", "T": "7",
        "l": "1", "L": "1",
    }
    return "".join(leet_map.get(c, c) for c in text)


def obfuscate_rot13(text: str) -> str:
    return codecs.encode(text, "rot_13")


def obfuscate_zero_width(text: str) -> str:
    return "\u200b".join(text)


def obfuscate_homoglyph(text: str) -> str:
    homoglyph_map = {
        "a": "\u0430", "e": "\u0435", "o": "\u043e",
        "p": "\u0440", "c": "\u0441", "x": "\u0445",
    }
    return "".join(homoglyph_map.get(c, c) for c in text)


OBFUSCATIONS = {
    "base64": obfuscate_base64,
    "leetspeak": obfuscate_leetspeak,
    "rot13": obfuscate_rot13,
    "zero_width": obfuscate_zero_width,
    "homoglyph": obfuscate_homoglyph,
}

import base64
import codecs


# ---------------------------------------------------------------------------
# Core evaluation
# ---------------------------------------------------------------------------

def method_scores(emb_test: np.ndarray,
                  ref_mal_raw: np.ndarray,
                  ref_safe_raw: np.ndarray,
                  dim: int,
                  seed: int) -> dict:
    """All methods for one (dataset, model, seed) given RAW test embeddings.

    Whitener(s) are fitted here on CLEAN calibration references only.
    For obfuscated runs emb_test holds obfuscated attack embeddings and the
    same clean-fitted whiteners are reused (deployment-realistic setting).
    """
    n_ref_mal = ref_mal_raw.shape[0]
    k = min(20, n_ref_mal)

    # Clean-fitted whiteners (Sigma_T on combined pool; Sigma_W pooled within)
    wh_t = fit_sigma_t_whitener(np.vstack([ref_mal_raw, ref_safe_raw]), dim)
    wh_w = PooledWithinClassWhitening(dim).fit(ref_mal_raw, ref_safe_raw)

    test_t = wh_t.transform(emb_test)
    mal_t = wh_t.transform(ref_mal_raw)
    safe_t = wh_t.transform(ref_safe_raw)

    test_w = wh_w.transform(emb_test)
    mal_w = wh_w.transform(ref_mal_raw)
    safe_w = wh_w.transform(ref_safe_raw)

    y_train = np.concatenate([np.ones(n_ref_mal), np.zeros(ref_safe_raw.shape[0])])
    lr = LogisticRegression(max_iter=1000, random_state=seed)
    lr.fit(np.vstack([ref_mal_raw, ref_safe_raw]), y_train)

    return {
        "A1_naive_cosine_raw": score_mean_direction(emb_test, ref_mal_raw),
        "A2_rsfi_svd_raw_k20": score_svd_subspace(emb_test, ref_mal_raw, k=k),
        "B1_discriminant_mean_raw": score_discriminant(emb_test, ref_mal_raw, ref_safe_raw),
        "B1b_SigmaT_wh": score_discriminant(test_t, mal_t, safe_t),
        "B1w_SigmaW_wh": score_discriminant(test_w, mal_w, safe_w),
        "C1_logreg_raw": lr.decision_function(emb_test),
    }


def evaluate_seed(embeddings: np.ndarray,
                  obf_embeddings: dict,
                  mal_idx_arr: np.ndarray,
                  safe_idx_arr: np.ndarray,
                  seed: int,
                  n_ref_mal: int,
                  n_ref_safe: int,
                  dim: int) -> list:
    """One leakage-free split per E2d/E8/E9 protocol; all obfuscation scenarios."""
    np.random.seed(seed)  # identical draw order as E2d/E8/E9
    ref_mal_idx = np.random.choice(mal_idx_arr, size=n_ref_mal, replace=False)
    ref_safe_idx = np.random.choice(safe_idx_arr, size=n_ref_safe, replace=False)

    test_mal_idx = np.setdiff1d(mal_idx_arr, ref_mal_idx)
    test_safe_idx = np.setdiff1d(safe_idx_arr, ref_safe_idx)

    row_positions_of_test_mal = np.nonzero(np.isin(mal_idx_arr, test_mal_idx))[0]
    assert len(row_positions_of_test_mal) == len(test_mal_idx)

    ref_mal_raw = embeddings[ref_mal_idx]
    ref_safe_raw = embeddings[ref_safe_idx]
    emb_test_safe_clean = embeddings[test_safe_idx]

    y_test = np.array(
        [1] * len(test_mal_idx) + [0] * len(test_safe_idx), dtype=int
    )
    base = dict(
        seed=seed,
        n_ref_mal=n_ref_mal, n_ref_safe=n_ref_safe,
        n_test_attack=len(test_mal_idx), n_test_safe=len(test_safe_idx),
    )

    rows = []
    for kind in ["clean"] + list(OBFUSCATIONS.keys()):
        if kind == "clean":
            emb_test_mal = embeddings[test_mal_idx]
        else:
            emb_test_mal = obf_embeddings[kind][row_positions_of_test_mal]

        emb_test = np.vstack([emb_test_mal, emb_test_safe_clean])
        scores = method_scores(emb_test, ref_mal_raw, ref_safe_raw, dim, seed)
        for m, sc in scores.items():
            rows.append(dict(
                base, obfuscation=kind, method=m,
                roc_auc=float(roc_auc_score(y_test, sc)),
                pr_auc=float(average_precision_score(y_test, sc)),
            ))
        print("    [{}/seed{}] ".format(kind, seed)
              + " ".join("{}={:.4f}".format(m.split("_")[0],
                                           roc_auc_score(y_test, sc))
                         for m, sc in scores.items()), flush=True)
    return rows


def run(datasets_filter=None, models_filter=None, sanity_gate=True):
    t_start = time.time()

    datasets = {
        "ToxicChat": load_toxicchat(),
        "Wild": load_wild(),
        "XSTest": load_xstest(),
    }
    if datasets_filter:
        datasets = {k: v for k, v in datasets.items() if k in datasets_filter}

    embedders = [
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]
    if models_filter:
        embedders = [m for m in embedders if any(f in m for f in models_filter)]

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
        print(f"Budget per class: n_ref_mal={n_ref_mal}, n_ref_safe={n_ref_safe}",
              flush=True)

        mal_idx_arr = np.where(labels == 1)[0]
        safe_idx_arr = np.where(labels == 0)[0]
        mal_texts = [texts[i] for i in mal_idx_arr]

        for model_id in embedders:
            model_short = model_id.split("/")[-1]
            print(f"\n--- Embedder: {model_short} ---", flush=True)
            embeddings = get_embeddings(texts, d_name, model_id)
            dim = embeddings.shape[1]

            # Pre-encode obfuscated variants of ALL malicious texts once
            obf_embeddings = {}
            for kind, fn in OBFUSCATIONS.items():
                obf_embeddings[kind] = get_embeddings(
                    [fn(t) for t in mal_texts],
                    f"{d_name}__obf-{kind}",
                    model_id,
                )

            for seed in range(N_SEEDS):
                seed_rows = evaluate_seed(
                    embeddings, obf_embeddings,
                    mal_idx_arr, safe_idx_arr,
                    seed=seed, n_ref_mal=n_ref_mal, n_ref_safe=n_ref_safe, dim=dim,
                )
                for r in seed_rows:
                    r.update(dataset=d_name, model=model_short, dim=dim)
                rows.extend(seed_rows)

    df = pd.DataFrame(rows)
    out_dir = Path(__file__).parent.parent / "data" / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / "E6b_obfuscation_boundary.csv"
    df.to_csv(out_csv, index=False)
    print(f"\nSaved: {out_csv} ({len(df)} rows, {time.time() - t_start:.0f}s)",
          flush=True)

    failures = sanity_check(df) if sanity_gate else []
    print_summary(df)

    if failures:
        print("\nSANITY GATE FAILURES (clean rows must reproduce committed CSVs):")
        for f in failures:
            print("  -", f)
        sys.exit(2)
    if sanity_gate:
        print("\nSANITY GATE PASSED: clean rows reproduce committed E8/E2d means.")


def sanity_check(df: pd.DataFrame, tol: float = 5e-4) -> list:
    """Clean rows must reproduce committed per-seed CSVs (mean & std of seeds)."""
    res_dir = Path(__file__).parent.parent / "data" / "results"

    committed_specs = {
        # method -> (csv filename, method-name-in-committed-csv)
        "B1_discriminant_mean_raw": ("E8_sigma_w.csv", "B1_raw"),
        "B1b_SigmaT_wh": ("E8_sigma_w.csv", "B1b_SigmaT_wh"),
        "B1w_SigmaW_wh": ("E8_sigma_w.csv", "B1w_SigmaW_wh"),
        "A1_naive_cosine_raw": ("E2d_safe_aware_multidataset.csv",
                                "A1_naive_cosine_raw"),
        "A2_rsfi_svd_raw_k20": ("E2d_safe_aware_multidataset.csv",
                                "A2_rsfi_svd_raw_k20"),
        "C1_logreg_raw": ("E2d_safe_aware_multidataset.csv", "C1_logreg_raw"),
    }

    failures = []
    clean = df[df.obfuscation == "clean"]
    for method, (fname, committed_name) in committed_specs.items():
        path = res_dir / fname
        if not path.exists():
            failures.append(f"{fname} missing -> cannot verify {method}")
            continue
        ref_df = pd.read_csv(path)
        mine = clean[clean.method == method]
        for (ds, mdl), g in ref_df.groupby(["dataset", "model"]):
            g_ref = g[g.method == committed_name]
            if g_ref.empty:
                continue
            exp_m, exp_s = g_ref.roc_auc.mean(), g_ref.roc_auc.std(ddof=1)
            m0 = mine[(mine.dataset == ds) & (mine.model == mdl)]
            if not ((mine.dataset == ds) & (mine.model == mdl)).any():
                continue  # this dataset/model was not part of the current run
            if m0.empty:
                failures.append(f"{method}/{ds}/{mdl}: rows absent in run output")
                continue
            got_m, got_s = m0.roc_auc.mean(), m0.roc_auc.std(ddof=1)
            for tag, a, b in (("mean", got_m, exp_m), ("std", got_s, exp_s)):
                if abs(a - b) > tol:
                    failures.append(
                        f"{method}/{ds}/{mdl}: clean {tag} {a:.4f} != "
                        f"committed {b:.4f} ({fname})"
                    )
    return failures


def print_summary(df: pd.DataFrame):
    piv = df.pivot_table(index=["dataset", "model", "method"],
                         columns="obfuscation", values="roc_auc")
    order = ["clean"] + list(OBFUSCATIONS.keys())
    piv = piv[[c for c in order if c in piv.columns]]
    with pd.option_context("display.max_columns", None, "display.width", 200):
        print("\n=== ROC-AUC means over seeds ===")
        print(piv.round(4).to_string())


if __name__ == "__main__":
    import argparse

    ap = argparse.ArgumentParser(description="E6b obfuscation boundary")
    ap.add_argument("--smoke", action="store_true",
                    help="quick pipeline check: XSTest + mpnet only, no gate")
    ap.add_argument("--datasets", default=None,
                    help="comma-separated subset, e.g. ToxicChat,Wild")
    ap.add_argument("--models", default=None,
                    help="comma-separated substrings, e.g. mpnet,bge-base")
    ap.add_argument("--no-gate", action="store_true")
    args = ap.parse_args()

    if args.smoke:
        run(datasets_filter=["XSTest"],
            models_filter=["all-mpnet-base-v2"], sanity_gate=False)
    else:
        run(datasets_filter=(args.datasets.split(",") if args.datasets else None),
            models_filter=(args.models.split(",") if args.models else None),
            sanity_gate=not args.no_gate)



