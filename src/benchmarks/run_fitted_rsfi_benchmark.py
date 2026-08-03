"""
Data-Driven RSFI Subspace Benchmark.
Extracts the k-dimensional threat subspace Q_k directly from principal components
of a small reference set of N_ref = 50 attack embeddings after ZCA whitening.
"""

import os
from pathlib import Path

CACHE_DIR = os.path.abspath("./hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

import argparse
import json
import sys
import time

import numpy as np
import pandas as pd
from sklearn.metrics import roc_auc_score, precision_recall_curve, auc
from sentence_transformers import SentenceTransformer

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsfi import RiemannianSphere, SphericalWhitening, MultiDimensionalRSFIFilter


def run_fitted_subspace_benchmark(
    model_name: str = "all-mpnet-base-v2", n_samples_per_class: int = 1000, k: int = 10
):
    print("=" * 90)
    print("   DATA-DRIVEN RSFI THREAT SUBSPACE FITTING & BENCHMARK")
    print("=" * 90 + "\n")

    model = SentenceTransformer(model_name, cache_folder=CACHE_DIR)
    dim = model.get_sentence_embedding_dimension()

    from rsfi.benchmarks.wildchat_10k import WildChatBenchmarkRunner

    runner = WildChatBenchmarkRunner(model_name=model_name, cache_folder=CACHE_DIR)
    samples = runner.load_dataset_samples(target_per_class=n_samples_per_class)

    np.random.seed(42)
    perm = np.random.permutation(len(samples))
    samples = [samples[i] for i in perm]
    y_true = np.array([1 if s.scenario_type == "MALICIOUS" else 0 for s in samples])

    # Separate reference training set (50 malicious prompts) vs test set
    mal_indices = np.where(y_true == 1)[0]
    safe_indices = np.where(y_true == 0)[0]

    ref_mal_idx = mal_indices[:50]
    ref_safe_idx = safe_indices[:50]
    test_idx = np.concatenate([mal_indices[50:], safe_indices[50:]])

    test_samples = [samples[i] for i in test_idx]
    y_test = y_true[test_idx]

    all_texts = [s.text for s in samples]
    print(f"[ENCODING] Encoding {len(all_texts)} prompts with {model_name}...")
    raw_embeddings = model.encode(
        all_texts, convert_to_numpy=True, batch_size=128, show_progress_bar=True
    )

    # Fit ZCA Whitening on Safe Reference Set + Calibration Corpus
    calib_corpus = [s.text for s in samples if s.scenario_type == "SAFE"][:500]
    calib_raw = model.encode(calib_corpus, convert_to_numpy=True)
    whitening = SphericalWhitening(dim=dim)
    whitening.fit(RiemannianSphere.normalize(calib_raw))

    # System Anchor S: Centroid of safe reference set after ZCA
    ref_safe_raw = raw_embeddings[ref_safe_idx]
    ref_safe_whitened = whitening.transform(ref_safe_raw)
    S = RiemannianSphere.normalize(np.mean(ref_safe_whitened, axis=0, keepdims=True))[0]

    # Data-Driven Threat Subspace Q_k: SVD / PCA of (Malicious_ref - S) in Tangent Space
    ref_mal_raw = raw_embeddings[ref_mal_idx]
    ref_mal_whitened = whitening.transform(ref_mal_raw)

    tangent_threats = []
    for i in range(len(ref_mal_whitened)):
        v_i = RiemannianSphere.log_map(S, ref_mal_whitened[i])
        tangent_threats.append(v_i)
    tangent_threats = np.array(tangent_threats)

    # Perform SVD on tangent threat vectors to get top-k principal threat directions
    U, Sigma, Vh = np.linalg.svd(tangent_threats, full_matrices=False)
    fitted_threat_vectors = Vh[:k]  # Top k threat basis vectors

    print(
        f"[SUBSPACE] Fitted top k={k} data-driven threat directions (explained variance: {np.sum(Sigma[:k]**2)/np.sum(Sigma**2)*100:.1f}%)"
    )

    # Build RSFI Filter with Fitted Subspace
    rsfi_fitted_filter = MultiDimensionalRSFIFilter(
        S, fitted_threat_vectors, alpha=1.5, beta=0.2, tau=0.5, is_tangent=True
    )

    # Evaluate on Test Set
    test_raw = raw_embeddings[test_idx]
    test_whitened = whitening.transform(test_raw)

    rsfi_scores = []
    latencies = []

    for i in range(len(test_whitened)):
        t0 = time.perf_counter()
        res = rsfi_fitted_filter.evaluate(test_whitened[i])
        lat = (time.perf_counter() - t0) * 1000.0

        rsfi_scores.append(-res["rsfi"])
        latencies.append(lat)

    rsfi_scores = np.array(rsfi_scores)
    auc_rsfi_fitted = float(roc_auc_score(y_test, rsfi_scores))

    # Baseline: Cosine to Mean Malicious Centroid
    mal_centroid = RiemannianSphere.normalize(
        np.mean(ref_mal_raw, axis=0, keepdims=True)
    )[0]
    cosine_scores = np.dot(RiemannianSphere.normalize(test_raw), mal_centroid)
    auc_cosine = float(roc_auc_score(y_test, cosine_scores))

    print("\n" + "=" * 90)
    print("     DATA-DRIVEN FITTED SUBSPACE BENCHMARK RESULTS")
    print("=" * 90)
    print(f"Model Backbone                          : {model_name} (d={dim})")
    print(f"Test Set Prompts (N_test)                : {len(y_test)}")
    print(f"Reference Train Set                      : 50 Safe + 50 Malicious Prompts")
    print("-" * 90)
    print(
        f"  1. RSFI Fitted k={k} Subspace (Our Method) : ROC-AUC = {auc_rsfi_fitted:.4f}"
    )
    print(f"  2. Baseline Cosine to Attack Centroid   : ROC-AUC = {auc_cosine:.4f}")
    print(
        f"  Mean RSFI Latency per prompt           : {np.mean(latencies)*1000.0:.1f} us ({np.mean(latencies):.3f} ms)"
    )
    print("=" * 90 + "\n")


if __name__ == "__main__":
    run_fitted_subspace_benchmark(
        model_name="all-mpnet-base-v2", n_samples_per_class=1000, k=10
    )
