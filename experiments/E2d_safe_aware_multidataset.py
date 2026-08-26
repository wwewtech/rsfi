"""
E2d_safe_aware_multidataset.py
================================================================================
Comprehensive Multi-Dataset & Multi-Embedder Safe-Aware Evaluation Suite

Evaluates:
1. Datasets: ToxicChat (homogeneous), Wild (heterogeneous), XSTest (contrastive)
2. Embedders: all-mpnet-base-v2 (768d), bge-base-en-v1.5 (768d), bge-large-en-v1.5 (1024d)
3. Methods:
   - Group A (Blind): A1 (naive cosine), A1b (whitened cosine), A2 (RSFI-SVD raw k20), A3 (RSFI-SVD whitened k20)
   - Group B (Safe-Aware): B1 (discriminant mean raw), B1b (discriminant mean whitened), B2 (contrastive SVD raw k20), B2b (contrastive SVD whitened k20)
   - Group C (Supervised Ceiling): C1 (LogReg raw), C1b (LogReg whitened)
4. Rigorous DeLong significance testing for key head-to-head pairs (B1b vs C1b, B1 vs A1, B1 vs C1, A3 vs A1).
"""

import sys
import os
from pathlib import Path
import numpy as np
import pandas as pd
import torch
from typing import Dict, List, Tuple
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import roc_auc_score, average_precision_score
from sentence_transformers import SentenceTransformer
from scipy import stats

sys.path.insert(0, str(Path(__file__).parent.parent / "src"))
from rsfi.whitening import SphericalWhitening

EPS = 1e-15
N_SEEDS = 5
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def l2norm(x: np.ndarray) -> np.ndarray:
    return x / (np.linalg.norm(x, axis=-1, keepdims=True) + EPS)


def fit_whitener(ref_emb: np.ndarray, dim: int) -> SphericalWhitening:
    wh = SphericalWhitening(dim=dim)
    wh.fit(ref_emb)
    return wh


# --- Methods ---
def score_mean_direction(emb_eval: np.ndarray, ref_mal_emb: np.ndarray) -> np.ndarray:
    mean = l2norm(ref_mal_emb.mean(axis=0, keepdims=True))
    return (l2norm(emb_eval) @ mean.T).flatten()


def score_svd_subspace(emb_eval: np.ndarray, ref_mal_emb: np.ndarray, k: int = 20) -> np.ndarray:
    ref_n = l2norm(ref_mal_emb)
    emb_n = l2norm(emb_eval)
    k_eff = min(k, ref_n.shape[0], ref_n.shape[1])
    U, S, Vt = np.linalg.svd(ref_n.T, full_matrices=False)
    U_k = U[:, :k_eff]
    proj = emb_n @ U_k
    return np.linalg.norm(proj, axis=1)


def score_discriminant_mean(emb_eval: np.ndarray, ref_mal_emb: np.ndarray, ref_safe_emb: np.ndarray) -> np.ndarray:
    direction = ref_mal_emb.mean(axis=0) - ref_safe_emb.mean(axis=0)
    direction = direction / (np.linalg.norm(direction) + EPS)
    return (l2norm(emb_eval) @ direction).flatten()


def score_contrastive_svd(emb_eval: np.ndarray, ref_mal_emb: np.ndarray, ref_safe_emb: np.ndarray, k: int = 20) -> np.ndarray:
    safe_mean = ref_safe_emb.mean(axis=0)
    displaced = ref_mal_emb - safe_mean
    k_eff = min(k, displaced.shape[0], displaced.shape[1])
    U, S, Vt = np.linalg.svd(displaced.T, full_matrices=False)
    U_k = U[:, :k_eff]
    proj = (emb_eval - safe_mean) @ U_k
    return np.linalg.norm(proj, axis=1)


# --- DeLong Test ---
def delong_test(y_true: np.ndarray, scores_1: np.ndarray, scores_2: np.ndarray) -> float:
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]
    n_pos, n_neg = len(pos_idx), len(neg_idx)
    if n_pos == 0 or n_neg == 0:
        return np.nan

    auc_1 = roc_auc_score(y_true, scores_1)
    auc_2 = roc_auc_score(y_true, scores_2)

    V_10_1 = np.zeros(n_pos)
    for i, pos_i in enumerate(pos_idx):
        comparisons = scores_1[pos_i] > scores_1[neg_idx]
        ties = scores_1[pos_i] == scores_1[neg_idx]
        V_10_1[i] = np.sum(comparisons) + 0.5 * np.sum(ties)
    V_10_1 /= n_neg

    V_01_1 = np.zeros(n_neg)
    for j, neg_j in enumerate(neg_idx):
        comparisons = scores_1[pos_idx] > scores_1[neg_j]
        ties = scores_1[pos_idx] == scores_1[neg_j]
        V_01_1[j] = np.sum(comparisons) + 0.5 * np.sum(ties)
    V_01_1 /= n_pos

    V_10_2 = np.zeros(n_pos)
    for i, pos_i in enumerate(pos_idx):
        comparisons = scores_2[pos_i] > scores_2[neg_idx]
        ties = scores_2[pos_i] == scores_2[neg_idx]
        V_10_2[i] = np.sum(comparisons) + 0.5 * np.sum(ties)
    V_10_2 /= n_neg

    V_01_2 = np.zeros(n_neg)
    for j, neg_j in enumerate(neg_idx):
        comparisons = scores_2[pos_idx] > scores_2[neg_j]
        ties = scores_2[pos_idx] == scores_2[neg_j]
        V_01_2[j] = np.sum(comparisons) + 0.5 * np.sum(ties)
    V_01_2 /= n_pos

    S_10 = np.var(V_10_1 - V_10_2, ddof=1) / n_pos
    S_01 = np.var(V_01_1 - V_01_2, ddof=1) / n_neg
    var_diff = S_10 + S_01
    if var_diff <= 0:
        return np.nan

    z = (auc_1 - auc_2) / np.sqrt(var_diff)
    return float(2 * (1 - stats.norm.cdf(abs(z))))


# --- Dataset Loaders ---
def load_toxicchat() -> Tuple[List[str], List[int]]:
    url = "https://huggingface.co/datasets/lmsys/toxic-chat/raw/main/data/0124/toxic-chat_annotation_train.csv"
    df = pd.read_csv(url)
    texts, labels = [], []
    for _, item in df.iterrows():
        text = str(item.get('user_input', ''))
        if not text or text == 'nan':
            continue
        tox = item.get('toxicity', 0)
        jb = item.get('jailbreaking', 0)
        if tox == 1 or jb == 1:
            labels.append(1)
            texts.append(text)
        elif tox == 0 and jb == 0:
            labels.append(0)
            texts.append(text)
    return texts, labels


def load_xstest() -> Tuple[List[str], List[int]]:
    url = "https://raw.githubusercontent.com/paul-rottger/xstest/main/xstest_prompts.csv"
    df = pd.read_csv(url)
    texts = df['prompt'].tolist()
    labels = (df['label'] == 'unsafe').astype(int).tolist()
    return texts, labels


def load_wild() -> Tuple[List[str], List[int]]:
    path = Path(__file__).parent.parent / "data" / "results" / "sfi_wild_10k_results.csv"
    df = pd.read_csv(path)
    texts = df['text'].tolist()
    if 'scenario_type' in df.columns:
        labels = (df['scenario_type'] == 'MALICIOUS').astype(int).tolist()
    else:
        labels = df['is_jailbreak'].astype(int).tolist()
    return texts, labels


# --- Embedding Manager ---
def get_embeddings(texts: List[str], dataset_name: str, model_id: str) -> np.ndarray:
    cache_dir = Path(__file__).parent.parent / "emb_cache"
    cache_dir.mkdir(parents=True, exist_ok=True)
    safe_model_name = model_id.replace('/', '_')
    cache_file = cache_dir / f"{dataset_name}_{safe_model_name}.npy"

    # Match pre-existing cache naming if applicable
    if dataset_name == "Wild":
        if "all-mpnet-base-v2" in model_id and (cache_dir / "all-mpnet-base-v2.npy").exists():
            return np.load(cache_dir / "all-mpnet-base-v2.npy")
        if "bge-base-en-v1.5" in model_id and (cache_dir / "BAAI_bge-base-en-v1.5.npy").exists():
            return np.load(cache_dir / "BAAI_bge-base-en-v1.5.npy")
        if "bge-large-en-v1.5" in model_id and (cache_dir / "BAAI_bge-large-en-v1.5.npy").exists():
            return np.load(cache_dir / "BAAI_bge-large-en-v1.5.npy")
        if "Qwen3" in model_id and (cache_dir / "Qwen_Qwen3-Embedding-8B.npy").exists():
            return np.load(cache_dir / "Qwen_Qwen3-Embedding-8B.npy")

    if cache_file.exists():
        print(f"  [Cache hit] {cache_file.name}")
        return np.load(cache_file)

    print(f"  [Encoding] {len(texts)} texts with {model_id} on {DEVICE}...")
    model = SentenceTransformer(model_id, device=DEVICE)
    emb = model.encode(texts, batch_size=128, convert_to_numpy=True, show_progress_bar=True)
    np.save(cache_file, emb)
    print(f"  [Cached] {cache_file.name}")
    return emb


def run_benchmark():
    print("=" * 80)
    print("E2d: COMPREHENSIVE MULTI-DATASET & MULTI-EMBEDDER SAFE-AWARE BENCHMARK")
    print("=" * 80)
    print(f"Device: {DEVICE}, Seeds: {N_SEEDS}\n")

    datasets = {
        "ToxicChat": load_toxicchat(),
        "Wild": load_wild(),
        "XSTest": load_xstest(),
    }

    embedders = [
        "sentence-transformers/all-mpnet-base-v2",
        "BAAI/bge-base-en-v1.5",
        "BAAI/bge-large-en-v1.5",
    ]

    all_results = []
    delong_results = []

    for d_name, (texts, labels) in datasets.items():
        labels = np.array(labels)
        n_mal = int(labels.sum())
        n_safe = len(labels) - n_mal
        print(f"\n{'#' * 80}\nDATASET: {d_name} ({len(texts)} items: {n_mal} malicious, {n_safe} safe)\n{'#' * 80}")

        # Reference budgets
        if n_mal < 250 or n_safe < 250:
            n_ref_mal = max(10, n_mal // 3)
            n_ref_safe = max(10, n_safe // 3)
        else:
            n_ref_mal = 200
            n_ref_safe = 200

        print(f"Budget per class: n_ref_mal={n_ref_mal}, n_ref_safe={n_ref_safe}")

        for model_id in embedders:
            model_short = model_id.split('/')[-1]
            print(f"\n--- Embedder: {model_short} ---")
            embeddings = get_embeddings(texts, d_name, model_id)
            dim = embeddings.shape[1]

            for seed in range(N_SEEDS):
                mal_idx = np.where(labels == 1)[0]
                safe_idx = np.where(labels == 0)[0]

                np.random.seed(seed)
                ref_mal_idx = np.random.choice(mal_idx, size=n_ref_mal, replace=False)
                ref_safe_idx = np.random.choice(safe_idx, size=n_ref_safe, replace=False)

                test_mal_idx = np.setdiff1d(mal_idx, ref_mal_idx)
                test_safe_idx = np.setdiff1d(safe_idx, ref_safe_idx)
                test_idx = np.concatenate([test_mal_idx, test_safe_idx])
                y_test = labels[test_idx]

                emb_test = embeddings[test_idx]
                ref_mal_raw = embeddings[ref_mal_idx]
                ref_safe_raw = embeddings[ref_safe_idx]

                # Combined whitening (2 * n_ref)
                ref_comb = np.vstack([ref_mal_raw, ref_safe_raw])
                wh = fit_whitener(ref_comb, dim=dim)
                emb_test_wh = wh.transform(emb_test)
                ref_mal_wh = wh.transform(ref_mal_raw)
                ref_safe_wh = wh.transform(ref_safe_raw)

                # Compute scores
                scores = {}
                # Blind
                scores["A1_naive_cosine_raw"] = score_mean_direction(emb_test, ref_mal_raw)
                scores["A1b_cosine_whitened"] = score_mean_direction(emb_test_wh, ref_mal_wh)
                scores["A2_rsfi_svd_raw_k20"] = score_svd_subspace(emb_test, ref_mal_raw, k=min(20, n_ref_mal))
                scores["A3_rsfi_svd_whitened_k20"] = score_svd_subspace(emb_test_wh, ref_mal_wh, k=min(20, n_ref_mal))

                # Safe-Aware
                scores["B1_discriminant_mean_raw"] = score_discriminant_mean(emb_test, ref_mal_raw, ref_safe_raw)
                scores["B1b_discriminant_mean_whitened"] = score_discriminant_mean(emb_test_wh, ref_mal_wh, ref_safe_wh)
                scores["B2_contrastive_svd_raw_k20"] = score_contrastive_svd(emb_test, ref_mal_raw, ref_safe_raw, k=min(20, n_ref_mal))
                scores["B2b_contrastive_svd_whitened_k20"] = score_contrastive_svd(emb_test_wh, ref_mal_wh, ref_safe_wh, k=min(20, n_ref_mal))

                # Supervised
                y_train = np.concatenate([np.ones(len(ref_mal_idx)), np.zeros(len(ref_safe_idx))])
                lr_raw = LogisticRegression(max_iter=1000, random_state=seed)
                lr_raw.fit(ref_comb, y_train)
                scores["C1_logreg_raw"] = lr_raw.decision_function(emb_test)

                lr_wh = LogisticRegression(max_iter=1000, random_state=seed)
                lr_wh.fit(np.vstack([ref_mal_wh, ref_safe_wh]), y_train)
                scores["C1b_logreg_whitened"] = lr_wh.decision_function(emb_test_wh)

                # Record AUC and PR-AUC
                for m_name, sc in scores.items():
                    auc = roc_auc_score(y_test, sc)
                    pr = average_precision_score(y_test, sc)
                    all_results.append({
                        "dataset": d_name,
                        "model": model_short,
                        "dim": dim,
                        "seed": seed,
                        "method": m_name,
                        "roc_auc": auc,
                        "pr_auc": pr,
                        "n_test": len(test_idx)
                    })

                # DeLong tests for key hypotheses
                pairs = [
                    ("B1b_discriminant_mean_whitened", "C1b_logreg_whitened", "B1b_vs_C1b"),
                    ("B1_discriminant_mean_raw", "A1_naive_cosine_raw", "B1_vs_A1"),
                    ("B1_discriminant_mean_raw", "C1_logreg_raw", "B1_vs_C1"),
                    ("A3_rsfi_svd_whitened_k20", "A1_naive_cosine_raw", "A3_vs_A1"),
                    ("B1b_discriminant_mean_whitened", "B1_discriminant_mean_raw", "B1b_vs_B1"),
                    ("B1b_discriminant_mean_whitened", "A1b_cosine_whitened", "B1b_vs_A1b"),
                ]
                for m1, m2, pair_name in pairs:
                    p_val = delong_test(y_test, scores[m1], scores[m2])
                    auc1 = roc_auc_score(y_test, scores[m1])
                    auc2 = roc_auc_score(y_test, scores[m2])
                    delong_results.append({
                        "dataset": d_name,
                        "model": model_short,
                        "seed": seed,
                        "pair": pair_name,
                        "method_1": m1,
                        "auc_1": auc1,
                        "method_2": m2,
                        "auc_2": auc2,
                        "auc_diff": auc1 - auc2,
                        "p_value": p_val
                    })

    # Save results
    res_df = pd.DataFrame(all_results)
    delong_df = pd.DataFrame(delong_results)

    out_res = Path(__file__).parent.parent / "data" / "results" / "E2d_safe_aware_multidataset.csv"
    out_del = Path(__file__).parent.parent / "data" / "results" / "E2d_delong_tests.csv"
    out_res.parent.mkdir(parents=True, exist_ok=True)
    res_df.to_csv(out_res, index=False)
    delong_df.to_csv(out_del, index=False)

    print(f"\n\nResults saved to {out_res}")
    print(f"DeLong tests saved to {out_del}")

    # Summary
    print("\n" + "=" * 100)
    print("OVERALL SUMMARY: ROC-AUC (mean +- std over 5 seeds)")
    print("=" * 100)
    pivot_auc = res_df.pivot_table(index=["dataset", "model"], columns="method", values="roc_auc", aggfunc=["mean"])
    with pd.option_context("display.max_columns", None, "display.width", 160):
        print(pivot_auc.round(4).to_string())

    print("\n" + "=" * 100)
    print("DELONG SIGNIFICANCE TEST SUMMARY: mean p-value & win count (AUC1 > AUC2)")
    print("=" * 100)
    summary_del = delong_df.groupby(["dataset", "model", "pair"]).agg(
        mean_diff=("auc_diff", "mean"),
        wins=("auc_diff", lambda x: int((x > 0).sum())),
        total=("auc_diff", "count"),
        mean_pval=("p_value", "mean")
    )
    print(summary_del.round(4).to_string())


if __name__ == "__main__":
    run_benchmark()
