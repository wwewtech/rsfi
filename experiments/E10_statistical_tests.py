"""
E10: Statistical Significance Tests
=====================================
Rigorous statistical validation of RSFI performance claims.

Tests implemented:
1. DeLong test for comparing AUC between methods
2. Bootstrap 95% confidence intervals
3. Multiple comparison correction (Holm method)
4. 10 random seeds for robustness

This provides p-values and confidence intervals for claims like
"RSFI significantly outperforms naive cosine" (or doesn't).
"""

import sys
from pathlib import Path
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple
from sklearn.metrics import roc_auc_score, average_precision_score
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sentence_transformers import SentenceTransformer
from scipy.stats import bootstrap
from scipy import stats

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening


def delong_test(y_true: np.ndarray, scores_1: np.ndarray, scores_2: np.ndarray) -> float:
    """
    DeLong test for comparing two AUC values.

    Returns p-value for H0: AUC_1 = AUC_2

    Implementation based on Sun & Xu (2014) "Fast Implementation of DeLong's Algorithm"
    """
    from sklearn.metrics import roc_auc_score

    # Compute AUCs
    auc_1 = roc_auc_score(y_true, scores_1)
    auc_2 = roc_auc_score(y_true, scores_2)

    # Separate positive and negative examples
    pos_idx = np.where(y_true == 1)[0]
    neg_idx = np.where(y_true == 0)[0]

    n_pos = len(pos_idx)
    n_neg = len(neg_idx)

    if n_pos == 0 or n_neg == 0:
        return np.nan

    # Compute structural components for method 1
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

    # Compute structural components for method 2
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

    # Compute covariance
    S_10 = np.var(V_10_1 - V_10_2, ddof=1) / n_pos
    S_01 = np.var(V_01_1 - V_01_2, ddof=1) / n_neg

    # Total variance
    var_diff = S_10 + S_01

    if var_diff <= 0:
        return np.nan

    # Z-statistic
    z = (auc_1 - auc_2) / np.sqrt(var_diff)

    # Two-tailed p-value
    p_value = 2 * (1 - stats.norm.cdf(abs(z)))

    return p_value


def bootstrap_ci(y_true: np.ndarray, scores: np.ndarray, n_resamples: int = 1000) -> Tuple[float, float, float]:
    """
    Compute bootstrap 95% CI for AUC.

    Returns: (mean_auc, ci_lower, ci_upper)
    """
    def auc_statistic(y, s):
        """Wrapper for bootstrap."""
        return roc_auc_score(y, s)

    # Bootstrap with paired resampling
    rng = np.random.default_rng(42)
    aucs = []

    for _ in range(n_resamples):
        idx = rng.choice(len(y_true), size=len(y_true), replace=True)
        y_boot = y_true[idx]
        s_boot = scores[idx]

        if len(np.unique(y_boot)) < 2:
            continue

        aucs.append(roc_auc_score(y_boot, s_boot))

    aucs = np.array(aucs)
    mean_auc = np.mean(aucs)
    ci_lower = np.percentile(aucs, 2.5)
    ci_upper = np.percentile(aucs, 97.5)

    return mean_auc, ci_lower, ci_upper


def compute_rsfi_scores(
    embeddings: np.ndarray,
    ref_indices: np.ndarray,
    k: int = 20,
    apply_whitening: bool = True
) -> np.ndarray:
    """Compute RSFI scores."""
    ref_emb = embeddings[ref_indices]

    if apply_whitening:
        wh = SphericalWhitening(dim=embeddings.shape[1])
        wh.fit(ref_emb)
        embeddings_wh = wh.transform(embeddings)
        ref_emb_wh = embeddings_wh[ref_indices]
    else:
        embeddings_wh = embeddings
        ref_emb_wh = ref_emb

    ref_emb_wh = ref_emb_wh / (np.linalg.norm(ref_emb_wh, axis=1, keepdims=True) + 1e-15)
    embeddings_wh = embeddings_wh / (np.linalg.norm(embeddings_wh, axis=1, keepdims=True) + 1e-15)

    U, S, Vt = np.linalg.svd(ref_emb_wh.T, full_matrices=False)
    U_k = U[:, :k]

    proj = embeddings_wh @ U_k
    scores = np.linalg.norm(proj, axis=1)

    return scores


def compute_naive_cosine_scores(
    embeddings: np.ndarray,
    ref_indices: np.ndarray
) -> np.ndarray:
    """Compute naive cosine similarity."""
    ref_emb = embeddings[ref_indices]
    ref_mean = np.mean(ref_emb, axis=0, keepdims=True)
    ref_mean = ref_mean / (np.linalg.norm(ref_mean) + 1e-15)

    embeddings_norm = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-15)
    scores = embeddings_norm @ ref_mean.T
    return scores.flatten()


def load_dataset() -> Tuple[List[str], List[int]]:
    """Load dataset for evaluation."""
    print("Loading dataset...")

    # Try wild dataset
    results_path = Path(__file__).parent.parent / "data" / "results" / "sfi_wild_10k_results.csv"
    if results_path.exists():
        df = pd.read_csv(results_path)
        texts = df['text'].tolist()
        labels = (df['scenario_type'] == 'MALICIOUS').astype(int).tolist()
        print(f"  Loaded wild dataset: {len(texts)} examples")
        return texts, labels

    # Try ToxicChat
    try:
        from datasets import load_dataset
        ds = load_dataset("lmsys/toxic-chat", "toxicchat0124")

        texts = []
        labels = []

        for item in ds['train']:
            text = item.get('user_input', '')
            if not text:
                continue

            toxicity = item.get('toxicity', 0)
            jailbreaking = item.get('jailbreaking', 0)

            if toxicity == 1 or jailbreaking == 1:
                labels.append(1)
            elif toxicity == 0 and jailbreaking == 0:
                labels.append(0)
            else:
                continue

            texts.append(text)

        print(f"  Loaded ToxicChat: {len(texts)} examples")
        return texts, labels

    except Exception as e:
        print(f"  Error loading dataset: {e}")
        return [], []


def main():
    """Run E10 statistical significance tests."""
    print("="*80)
    print("E10: STATISTICAL SIGNIFICANCE TESTS")
    print("="*80)

    # Configuration
    MODEL_NAME = "sentence-transformers/all-mpnet-base-v2"
    N_REF = 200
    N_VAL = 200
    N_SEEDS = 10  # More seeds for robust statistics

    print(f"\nConfiguration:")
    print(f"  Model: {MODEL_NAME}")
    print(f"  N_ref: {N_REF}")
    print(f"  N_val: {N_VAL}")
    print(f"  N_seeds: {N_SEEDS}")

    # Load model
    print("\nLoading embedding model...")
    model = SentenceTransformer(MODEL_NAME)

    # Load dataset
    texts, labels = load_dataset()
    if len(texts) == 0:
        print("\nERROR: No dataset loaded.")
        return

    # Encode all texts
    print("\nEncoding texts...")
    embeddings = model.encode(texts, convert_to_numpy=True, show_progress_bar=True)
    labels = np.array(labels)

    # Store all scores for statistical tests
    all_scores = {
        'rsfi': [],
        'cosine': [],
        'logreg': []
    }
    all_y_test = []

    # Run across seeds
    for seed in range(N_SEEDS):
        print(f"\nSeed {seed+1}/{N_SEEDS}")

        # Split data
        malicious_idx = np.where(labels == 1)[0]
        safe_idx = np.where(labels == 0)[0]

        if len(malicious_idx) < N_REF + 100 or len(safe_idx) < N_VAL + 100:
            print("  WARNING: Not enough data")
            continue

        np.random.seed(seed)
        ref_idx = np.random.choice(malicious_idx, size=N_REF, replace=False)
        remaining_mal_idx = np.setdiff1d(malicious_idx, ref_idx)

        val_mal_idx = np.random.choice(remaining_mal_idx, size=N_VAL//2, replace=False)
        val_safe_idx = np.random.choice(safe_idx, size=N_VAL//2, replace=False)
        val_idx = np.concatenate([val_mal_idx, val_safe_idx])

        test_idx = np.setdiff1d(np.arange(len(texts)), np.concatenate([ref_idx, val_idx]))

        y_test = labels[test_idx]
        all_y_test.append(y_test)

        # Combine for scoring
        all_idx = np.concatenate([ref_idx, test_idx])

        # RSFI-SVD
        scores_rsfi = compute_rsfi_scores(
            embeddings[all_idx],
            np.arange(N_REF),
            k=20,
            apply_whitening=True
        )[N_REF:]
        all_scores['rsfi'].append(scores_rsfi)

        # Naive cosine
        scores_cosine = compute_naive_cosine_scores(
            embeddings[all_idx],
            np.arange(N_REF)
        )[N_REF:]
        all_scores['cosine'].append(scores_cosine)

        # LogReg
        logreg = LogisticRegression(max_iter=1000, random_state=seed)
        train_idx_local = np.concatenate([
            np.arange(N_REF),
            N_REF + np.arange(len(val_idx))
        ])
        X_all = np.vstack([embeddings[ref_idx], embeddings[val_idx], embeddings[test_idx]])
        y_all = np.concatenate([labels[ref_idx], labels[val_idx], y_test])

        logreg.fit(X_all[train_idx_local], y_all[train_idx_local])
        scores_logreg = logreg.decision_function(X_all[N_REF+len(val_idx):])
        all_scores['logreg'].append(scores_logreg)

        print(f"  AUCs: RSFI={roc_auc_score(y_test, scores_rsfi):.4f}, "
              f"Cosine={roc_auc_score(y_test, scores_cosine):.4f}, "
              f"LogReg={roc_auc_score(y_test, scores_logreg):.4f}")

    # Statistical tests
    print("\n" + "="*80)
    print("STATISTICAL TESTS")
    print("="*80)

    results = []

    # For each seed, compute DeLong test between methods
    pvals_rsfi_vs_cosine = []
    pvals_rsfi_vs_logreg = []
    pvals_cosine_vs_logreg = []

    for i in range(len(all_y_test)):
        y = all_y_test[i]

        # RSFI vs Cosine
        p = delong_test(y, all_scores['rsfi'][i], all_scores['cosine'][i])
        if not np.isnan(p):
            pvals_rsfi_vs_cosine.append(p)

        # RSFI vs LogReg
        p = delong_test(y, all_scores['rsfi'][i], all_scores['logreg'][i])
        if not np.isnan(p):
            pvals_rsfi_vs_logreg.append(p)

        # Cosine vs LogReg
        p = delong_test(y, all_scores['cosine'][i], all_scores['logreg'][i])
        if not np.isnan(p):
            pvals_cosine_vs_logreg.append(p)

    # Combine all scores across seeds for overall CI
    all_y_combined = np.concatenate(all_y_test)
    all_rsfi_combined = np.concatenate(all_scores['rsfi'])
    all_cosine_combined = np.concatenate(all_scores['cosine'])
    all_logreg_combined = np.concatenate(all_scores['logreg'])

    # Bootstrap CIs
    print("\nBootstrap 95% Confidence Intervals:")
    rsfi_mean, rsfi_lower, rsfi_upper = bootstrap_ci(all_y_combined, all_rsfi_combined)
    print(f"  RSFI-SVD: {rsfi_mean:.4f} [{rsfi_lower:.4f}, {rsfi_upper:.4f}]")

    cosine_mean, cosine_lower, cosine_upper = bootstrap_ci(all_y_combined, all_cosine_combined)
    print(f"  Naive cosine: {cosine_mean:.4f} [{cosine_lower:.4f}, {cosine_upper:.4f}]")

    logreg_mean, logreg_lower, logreg_upper = bootstrap_ci(all_y_combined, all_logreg_combined)
    print(f"  LogReg: {logreg_mean:.4f} [{logreg_lower:.4f}, {logreg_upper:.4f}]")

    # DeLong tests
    print("\nDeLong Test p-values (mean across seeds):")
    print(f"  RSFI vs Cosine: p = {np.mean(pvals_rsfi_vs_cosine):.4f}")
    print(f"  RSFI vs LogReg: p = {np.mean(pvals_rsfi_vs_logreg):.4f}")
    print(f"  Cosine vs LogReg: p = {np.mean(pvals_cosine_vs_logreg):.4f}")

    # Holm correction for multiple comparisons
    from statsmodels.stats.multitest import multipletests

    all_pvals = pvals_rsfi_vs_cosine + pvals_rsfi_vs_logreg + pvals_cosine_vs_logreg
    if len(all_pvals) > 0:
        reject, pvals_corrected, _, _ = multipletests(all_pvals, method='holm')
        print("\nHolm-corrected p-values:")
        n = len(pvals_rsfi_vs_cosine)
        print(f"  RSFI vs Cosine: {np.mean(pvals_corrected[:n]):.4f}")
        print(f"  RSFI vs LogReg: {np.mean(pvals_corrected[n:2*n]):.4f}")
        print(f"  Cosine vs LogReg: {np.mean(pvals_corrected[2*n:]):.4f}")

    # Save results
    summary_df = pd.DataFrame([
        {
            'method': 'RSFI-SVD',
            'auc_mean': rsfi_mean,
            'auc_ci_lower': rsfi_lower,
            'auc_ci_upper': rsfi_upper,
            'ci_width': rsfi_upper - rsfi_lower
        },
        {
            'method': 'naive_cosine',
            'auc_mean': cosine_mean,
            'auc_ci_lower': cosine_lower,
            'auc_ci_upper': cosine_upper,
            'ci_width': cosine_upper - cosine_lower
        },
        {
            'method': 'LogReg',
            'auc_mean': logreg_mean,
            'auc_ci_lower': logreg_lower,
            'auc_ci_upper': logreg_upper,
            'ci_width': logreg_upper - logreg_lower
        }
    ])

    output_path = Path(__file__).parent.parent / "results" / "E10_statistical_tests.csv"
    output_path.parent.mkdir(parents=True, exist_ok=True)
    summary_df.to_csv(output_path, index=False)

    print(f"\n\nResults saved to: {output_path}")

    print("\n" + "="*80)
    print("INTERPRETATION")
    print("="*80)
    print("p < 0.05: Statistically significant difference")
    print("p ≥ 0.05: No significant difference (could be due to noise)")
    print("CI width: Narrower = more consistent performance")


if __name__ == "__main__":
    main()
