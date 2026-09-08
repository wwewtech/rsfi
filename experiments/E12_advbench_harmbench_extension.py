"""
E12_advbench_harmbench_extension.py
================================================================================
4th-block data extension of the per-seed reproducible protocol (E2d + E8) to
two NEW dataset pairs built from canonical adversarial-behavior corpora:

  1. "AdvBench"  : 520 harmful behaviors (Zou et al., 2023, arXiv:2307.15043,
                    canonical copy from github.com/llm-attacks/llm-attacks)
                    + 400 safe imperative instructions (Stanford Alpaca,
                    Taori et al., 2023; self-contained, no input field).
  2. "HarmBench" : 400 standard text behaviors (Mazeika et al., 2024,
                    arXiv:2402.05793, functional + contextual,
                    github.com/centerforaisafety/HarmBench)
                    + 400 safe imperative instructions (Alpaca, disjoint pool).

Design notes (documented choices):
  - Harmless classes of AdvBench/HarmBench are NOT published by the original
    authors; third-party mirrors mix corpora or add chat wrappers. We pair
    each malicious corpus with Alpaca imperative instructions instead, which
    is the standard negative-pool construction for behavior-level guardrail
    evaluation (style-homogeneous with the malicious imperatives).
  - Alpaca safe pools are DISJOINT between the two datasets (one fixed
    RandomState permutation: first 400 -> AdvBench, next 400 -> HarmBench of
    deduplicated self-contained instructions), so the two benchmarks share
    no calibration text.
  - Splits, budgets, methods, seeds and DeLong pairs are replicated verbatim
    from E2d_safe_aware_multidataset.py (battery A/B/C) and
    E8_sigma_w_whitening.py (Sigma_W block). No method code is re-implemented.
  - Qwen3-Embedding-8B embeddings are encoded with the exact E2e fallback
    encoder (last non-padding token, bf16, device_map=auto) and cached under
    emb_cache/{dataset}_Qwen_Qwen3-Embedding-8B.npy.

Outputs (data/results/):
  E12_advbench_harmbench_e2d.csv            (schema = E2d results + E8 group B)
  E12_advbench_harmbench_delong.csv         (union of E2d + E8 delong pairs)
  E12_advbench_harmbench_delong_sigma_w.csv (E8-style pairs only)
  E12_mode_diagnostics.csv                  (per dataset x embedder geometry)
"""

import sys
import json
from pathlib import Path

import numpy as np
import pandas as pd
import torch
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    l2norm, fit_whitener,
    score_mean_direction, score_svd_subspace, score_discriminant_mean,
    score_contrastive_svd, delong_test, N_SEEDS,
)
from E8_sigma_w_whitening import (  # noqa: E402
    fit_sigma_t_whitener, PooledWithinClassWhitening, score_discriminant,
)
from E2e_qwen_extension import _encode_qwen_manual  # noqa: E402

ROOT = Path(__file__).parent.parent
RAW = ROOT / "data" / "raw"
EMB_CACHE = ROOT / "emb_cache"
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

EMBEDDERS = [
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
    "Qwen/Qwen3-Embedding-8B",
]

# Fixed, documented safe-pool construction: one deterministic permutation,
# first 400 instructions -> AdvBench safe pool, next 400 -> HarmBench pool.
ALPACA_SEED = 20260908
SAFE_POOL_ADV = 400
SAFE_POOL_HB = 400
MIN_INSTR_LEN = 15

# --------------------------------------------------------------------------
# Dataset assembly
# --------------------------------------------------------------------------

def load_alpaca_selfcontained() -> list:
    """Deduplicated self-contained Alpaca instructions (no input field)."""
    data = json.load(open(RAW / "alpaca_data.json", encoding="utf-8"))
    seen, pool = set(), []
    for r in data:
        ins = r["instruction"].strip()
        if ins and not r["input"].strip() and len(ins) >= MIN_INSTR_LEN \
                and ins not in seen:
            seen.add(ins)
            pool.append(ins)
    return pool


def build_behavior_datasets():
    """Returns {dataset_name: (texts, labels)} with fixed disjoint safe pools."""
    adv = pd.read_csv(RAW / "advbench_harmful_behaviors.csv")
    hb = pd.read_csv(RAW / "harmbench_all.csv")

    adv_mal = adv["goal"].astype(str).tolist()
    hb_mal = hb["Behavior"].astype(str).tolist()

    pool = load_alpaca_selfcontained()
    order = np.random.RandomState(ALPACA_SEED).permutation(len(pool))
    safe_adv = [pool[i] for i in order[:SAFE_POOL_ADV]]
    safe_hb = [pool[i] for i in order[SAFE_POOL_ADV:SAFE_POOL_ADV + SAFE_POOL_HB]]

    datasets = {
        "AdvBench": (adv_mal + safe_adv, [1] * len(adv_mal) + [0] * len(safe_adv)),
        "HarmBench": (hb_mal + safe_hb, [1] * len(hb_mal) + [0] * len(safe_hb)),
    }
    for name, (texts, labels) in datasets.items():
        print(f"  [{name}] {len(texts)} items: "
              f"{sum(labels)} malicious / {len(labels) - sum(labels)} safe")
    return datasets


# --------------------------------------------------------------------------
# Embedding cache (standard models via sentence-transformers, Qwen via E2e)
# --------------------------------------------------------------------------

def get_embeddings_any(texts, dataset_name: str, model_id: str) -> np.ndarray:
    if "Qwen3" in model_id:
        cache_file = EMB_CACHE / f"{dataset_name}_Qwen_Qwen3-Embedding-8B.npy"
        if cache_file.exists():
            emb = np.load(cache_file)
            assert emb.shape[0] == len(texts)
            print(f"  [Cache hit] {cache_file.name} {emb.shape}")
            return emb
        print(f"  [Encoding] {len(texts)} texts with Qwen3-Embedding-8B ...")
        emb = _encode_qwen_manual(texts)
        EMB_CACHE.mkdir(parents=True, exist_ok=True)
        np.save(cache_file, emb)
        print(f"  [Cached] {cache_file.name}")
        return emb

    from sentence_transformers import SentenceTransformer
    safe_name = model_id.replace("/", "_")
    cache_file = EMB_CACHE / f"{dataset_name}_{safe_name}.npy"
    if cache_file.exists():
        emb = np.load(cache_file)
        assert emb.shape[0] == len(texts)
        print(f"  [Cache hit] {cache_file.name} {emb.shape}")
        return emb
    print(f"  [Encoding] {len(texts)} texts with {model_id} on {DEVICE} ...")
    model = SentenceTransformer(model_id, device=DEVICE)
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True,
                       show_progress_bar=True)
    np.save(cache_file, emb)
    return emb

# --------------------------------------------------------------------------
# Mode diagnostics (geometry of the malicious vs safe class in RAW space)
# --------------------------------------------------------------------------

def mode_diagnostics(ref_mal_raw, ref_safe_raw) -> dict:
    m = l2norm(ref_mal_raw)
    s = l2norm(ref_safe_raw)
    mu_m = m.mean(axis=0)
    mu_s = s.mean(axis=0)
    mu_m = mu_m / (np.linalg.norm(mu_m) + 1e-15)
    mu_s = mu_s / (np.linalg.norm(mu_s) + 1e-15)

    def mean_pairwise_cos(x):
        n = x.shape[0]
        tot, cnt = 0.0, 0
        for i in range(0, n, 64):
            blk = x[i:i + 64] @ x.T
            sq = blk.shape[0]
            tot += blk.sum() - np.trace(blk[:sq])
            cnt += blk.size - sq
        return float(tot / max(cnt, 1))

    return {
        "cos_mu_mal_safe": float(mu_m @ mu_s),
        "mal_homog": mean_pairwise_cos(m),
        "safe_homog": mean_pairwise_cos(s),
    }


# --------------------------------------------------------------------------
# One (dataset, model, seed): full E2d battery + E8 Sigma_W block
# --------------------------------------------------------------------------

def run_battery(d_name, embeddings, labels, seed, n_ref_mal, n_ref_safe,
                mal_idx, safe_idx):
    rng = np.random.RandomState(seed)
    ref_mal_idx = rng.choice(mal_idx, size=n_ref_mal, replace=False)
    ref_safe_idx = rng.choice(safe_idx, size=n_ref_safe, replace=False)

    test_mal_idx = np.setdiff1d(mal_idx, ref_mal_idx)
    test_safe_idx = np.setdiff1d(safe_idx, ref_safe_idx)
    test_idx = np.concatenate([test_mal_idx, test_safe_idx])
    y_test = labels[test_idx]

    emb_test = embeddings[test_idx]
    ref_mal_raw = embeddings[ref_mal_idx]
    ref_safe_raw = embeddings[ref_safe_idx]
    ref_comb = np.vstack([ref_mal_raw, ref_safe_raw])
    dim = embeddings.shape[1]
    k = min(20, n_ref_mal)

    res, dlong, diag_rows = [], [], []
    base = {"dataset": d_name, "seed": seed, "n_test": len(test_idx)}

    # --- E2d battery (A/B/C) ----------------------------------------------
    wh = fit_whitener(ref_comb, dim=dim)
    emb_test_wh = wh.transform(emb_test)
    ref_mal_wh = wh.transform(ref_mal_raw)
    ref_safe_wh = wh.transform(ref_safe_raw)

    y_train = np.concatenate([np.ones(n_ref_mal), np.zeros(n_ref_safe)])
    lr_raw = LogisticRegression(max_iter=1000, random_state=seed)
    lr_raw.fit(ref_comb, y_train)
    lr_wh = LogisticRegression(max_iter=1000, random_state=seed)
    lr_wh.fit(np.vstack([ref_mal_wh, ref_safe_wh]), y_train)

    scores = {
        "A1_naive_cosine_raw": score_mean_direction(emb_test, ref_mal_raw),
        "A1b_cosine_whitened": score_mean_direction(emb_test_wh, ref_mal_wh),
        "A2_rsfi_svd_raw_k20": score_svd_subspace(emb_test, ref_mal_raw, k=k),
        "A3_rsfi_svd_whitened_k20": score_svd_subspace(emb_test_wh, ref_mal_wh, k=k),
        "B1_discriminant_mean_raw": score_discriminant_mean(emb_test, ref_mal_raw, ref_safe_raw),
        "B1b_discriminant_mean_whitened": score_discriminant_mean(emb_test_wh, ref_mal_wh, ref_safe_wh),
        "B2_contrastive_svd_raw_k20": score_contrastive_svd(emb_test, ref_mal_raw, ref_safe_raw, k=k),
        "B2b_contrastive_svd_whitened_k20": score_contrastive_svd(emb_test_wh, ref_mal_wh, ref_safe_wh, k=k),
        "C1_logreg_raw": lr_raw.decision_function(emb_test),
        "C1b_logreg_whitened": lr_wh.decision_function(emb_test_wh),
    }
    for m_name, sc in scores.items():
        res.append(dict(base, model=None, method=m_name,
                        roc_auc=roc_auc_score(y_test, sc),
                        pr_auc=average_precision_score(y_test, sc)))

    for m1, m2, tag in [
        ("B1b_discriminant_mean_whitened", "C1b_logreg_whitened", "B1b_vs_C1b"),
        ("B1_discriminant_mean_raw", "A1_naive_cosine_raw", "B1_vs_A1"),
        ("B1_discriminant_mean_raw", "C1_logreg_raw", "B1_vs_C1"),
        ("A3_rsfi_svd_whitened_k20", "A1_naive_cosine_raw", "A3_vs_A1"),
        ("B1b_discriminant_mean_whitened", "B1_discriminant_mean_raw", "B1b_vs_B1"),
        ("B1b_discriminant_mean_whitened", "A1b_cosine_whitened", "B1b_vs_A1b"),
    ]:
        dlong.append(dict(base, model=None, pair=tag, method_1=m1, method_2=m2,
                          auc_1=roc_auc_score(y_test, scores[m1]),
                          auc_2=roc_auc_score(y_test, scores[m2]),
                          auc_diff=float(roc_auc_score(y_test, scores[m1])
                                         - roc_auc_score(y_test, scores[m2])),
                          p_value=delong_test(y_test, scores[m1], scores[m2])))

    # --- E8 Sigma_W block ---------------------------------------------------
    wh_t = fit_sigma_t_whitener(ref_comb, dim)
    wh_w = PooledWithinClassWhitening(dim).fit(ref_mal_raw, ref_safe_raw)

    test_t = wh_t.transform(emb_test)
    mal_t = wh_t.transform(ref_mal_raw)
    safe_t = wh_t.transform(ref_safe_raw)
    test_w = wh_w.transform(emb_test)
    mal_w = wh_w.transform(ref_mal_raw)
    safe_w = wh_w.transform(ref_safe_raw)

    scores_b = {
        "B1_raw": score_discriminant(emb_test, ref_mal_raw, ref_safe_raw),
        "B1b_SigmaT_wh": score_discriminant(test_t, mal_t, safe_t),
        "B1w_SigmaW_wh": score_discriminant(test_w, mal_w, safe_w),
    }
    for m_name, sc in scores_b.items():
        res.append(dict(base, model=None, method=m_name,
                        roc_auc=roc_auc_score(y_test, sc),
                        pr_auc=average_precision_score(y_test, sc)))

    for m1, m2, tag in [
        ("B1w_SigmaW_wh", "B1b_SigmaT_wh", "B1w_vs_B1b"),
        ("B1b_SigmaT_wh", "B1_raw", "B1b_vs_B1"),
        ("B1w_SigmaW_wh", "B1_raw", "B1w_vs_B1"),
    ]:
        dlong.append(dict(base, model=None, pair=tag, method_1=m1, method_2=m2,
                          auc_1=roc_auc_score(y_test, scores_b[m1]),
                          auc_2=roc_auc_score(y_test, scores_b[m2]),
                          auc_diff=float(roc_auc_score(y_test, scores_b[m1])
                                         - roc_auc_score(y_test, scores_b[m2])),
                          p_value=delong_test(y_test, scores_b[m1], scores_b[m2])))

    diag_rows.append(dict(base, model=None,
                          **mode_diagnostics(ref_mal_raw, ref_safe_raw)))
    return res, dlong, diag_rows


def main():
    print("=" * 80)
    print("E12: ADVBENCH + HARMBENCH EXTENSION (E2d battery + E8 Sigma_W block)")
    print("=" * 80)
    print(f"Device: {DEVICE}, seeds: {N_SEEDS}")

    datasets = build_behavior_datasets()
    res_all, delong_all, diag_all = [], [], []

    for d_name, (texts, labels) in datasets.items():
        labels_arr = np.array(labels)
        mal_idx = np.where(labels_arr == 1)[0]
        safe_idx = np.where(labels_arr == 0)[0]
        n_ref_mal = n_ref_safe = 200  # E2d budget rule (both classes >= 250)

        for model_id in EMBEDDERS:
            model_short = model_id.split("/")[-1]
            print(f"\n--- {d_name} x {model_short} ---", flush=True)
            embeddings = get_embeddings_any(texts, d_name, model_id)

            for seed in range(N_SEEDS):
                r, d, g = run_battery(
                    d_name, embeddings, labels_arr, seed,
                    n_ref_mal, n_ref_safe, mal_idx, safe_idx)
                for row in r:
                    row["model"] = model_short
                for row in d:
                    row["model"] = model_short
                for row in g:
                    row["model"] = model_short
                res_all.extend(r)
                delong_all.extend(d)
                diag_all.extend(g)
                b1 = next(x["roc_auc"] for x in r
                          if x["method"] == "B1_discriminant_mean_raw")
                b1w = next(x["roc_auc"] for x in r
                           if x["method"] == "B1w_SigmaW_wh")
                a1 = next(x["roc_auc"] for x in r
                          if x["method"] == "A1_naive_cosine_raw")
                print(f"  seed {seed}: B1={b1:.4f} B1w={b1w:.4f} A1={a1:.4f}",
                      flush=True)

    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res_all).to_csv(out / "E12_advbench_harmbench_e2d.csv", index=False)
    df_dl = pd.DataFrame(delong_all)
    df_dl.to_csv(out / "E12_advbench_harmbench_delong.csv", index=False)
    mask = df_dl["pair"].isin(["B1w_vs_B1b", "B1b_vs_B1", "B1w_vs_B1"])
    df_dl[mask].to_csv(out / "E12_advbench_harmbench_delong_sigma_w.csv",
                       index=False)
    pd.DataFrame(diag_all).to_csv(out / "E12_mode_diagnostics.csv", index=False)

    df = pd.DataFrame(res_all)
    print("\n=== ROC-AUC (mean over seeds) ===")
    piv = df.pivot_table(index=["dataset", "model"], columns="method",
                         values="roc_auc")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(piv.round(4).to_string())
    print("\nSaved 4 CSV files to", out)


if __name__ == "__main__":
    main()
