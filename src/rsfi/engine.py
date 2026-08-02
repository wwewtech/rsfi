"""
Legacy / High-level Benchmark Engine for RSFI.
"""

from dataclasses import dataclass, asdict
import logging
import time
from typing import Any, Dict, List, Tuple

import numpy as np
from sentence_transformers import SentenceTransformer

from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening


@dataclass
class SentenceTelemetry:
    scenario_id: str
    scenario_type: str  # "SAFE" or "MALICIOUS"
    sentence_index: int
    text: str
    proj_system: float
    proj_threat: float
    sfi_score: float
    naive_cosine_sys: float
    naive_cosine_thr: float
    is_blocked: bool
    latency_ms: float


@dataclass
class BenchmarkSummary:
    total_sentences: int
    true_positives: int
    true_negatives: int
    false_positives: int
    false_negatives: int
    accuracy: float
    precision: float
    recall: float
    f1_score: float
    mean_latency_ms: float


class ProductionSFIEngine:
    def __init__(self, model_name: str = 'sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2'):
        logging.info(f"Initializing RSFI Engine with model: {model_name}")
        self.model_name = model_name
        self.model = SentenceTransformer(model_name)
        self.mean_vector: np.ndarray = None

    def fit_calibration_corpus(self, corpus: List[str]) -> None:
        """Compute empirical mean vector for anisotropy removal."""
        logging.info(f"Calibrating centering on corpus of size {len(corpus)}...")
        embeddings = self.model.encode(corpus, convert_to_numpy=True)
        self.mean_vector = np.mean(embeddings, axis=0)

    def _preprocess_vector(self, raw_vector: np.ndarray) -> np.ndarray:
        """Centering and L2-normalization."""
        vector = raw_vector.copy()
        if self.mean_vector is not None:
            vector -= self.mean_vector
        return RiemannianSphere.normalize(vector)

    def construct_orthogonal_basis(
        self, system_prompt: str, threat_anchor: str
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Construct Gram-Schmidt orthogonal basis (e1, e2)."""
        raw_sys = self.model.encode(system_prompt, convert_to_numpy=True)
        raw_thr = self.model.encode(threat_anchor, convert_to_numpy=True)

        s = self._preprocess_vector(raw_sys)
        t = self._preprocess_vector(raw_thr)

        # Gram-Schmidt Orthogonalization
        e1 = s
        u2 = t - np.dot(t, e1) * e1
        norm_u2 = np.linalg.norm(u2)
        e2 = u2 / norm_u2 if norm_u2 > 0 else u2

        return e1, e2, raw_sys, raw_thr

    def evaluate_sentence(
        self,
        sentence: str,
        e1: np.ndarray,
        e2: np.ndarray,
        raw_sys: np.ndarray,
        raw_thr: np.ndarray,
    ) -> Tuple[float, float, float, float, float, float]:
        """Audit a single sentence using orthogonal projection vs naive cosine."""
        t0 = time.perf_counter()

        raw_r = self.model.encode(sentence, convert_to_numpy=True)
        r = self._preprocess_vector(raw_r)

        # Orthogonal projections
        p_sys = float(np.dot(r, e1))
        p_thr = float(np.dot(r, e2))
        sfi = p_sys - p_thr

        # Naive cosine similarity without orthogonalization
        naive_sys = float(np.dot(raw_r, raw_sys) / (np.linalg.norm(raw_r) * np.linalg.norm(raw_sys)))
        naive_thr = float(np.dot(raw_r, raw_thr) / (np.linalg.norm(raw_r) * np.linalg.norm(raw_thr)))

        latency_ms = (time.perf_counter() - t0) * 1000.0

        return sfi, p_sys, p_thr, naive_sys, naive_thr, latency_ms


class SFIBenchmarkRunner:
    def __init__(self, engine: ProductionSFIEngine, threshold: float = 0.0):
        self.engine = engine
        self.threshold = threshold
        self.telemetry_logs: List[SentenceTelemetry] = []

    def run_suite(self, scenarios: List[Dict[str, Any]], system_prompt: str, threat_anchor: str):
        e1, e2, raw_sys, raw_thr = self.engine.construct_orthogonal_basis(system_prompt, threat_anchor)

        print("\n" + "=" * 80)
        print(f"START SFI BENCHMARK | Threshold: {self.threshold} | Model: {self.engine.model_name}")
        print("=" * 80 + "\n")

        for sc in scenarios:
            sc_id = sc["id"]
            sc_type = sc["type"]
            sentences = sc["stream"]

            print(f"► Scenario: [{sc_id}] ({sc_type}) — {sc['description']}")

            for idx, text in enumerate(sentences):
                sfi, p_sys, p_thr, n_sys, n_thr, lat = self.engine.evaluate_sentence(
                    text, e1, e2, raw_sys, raw_thr
                )

                is_blocked = sfi < self.threshold

                log_entry = SentenceTelemetry(
                    scenario_id=sc_id,
                    scenario_type=sc_type,
                    sentence_index=idx + 1,
                    text=text,
                    proj_system=p_sys,
                    proj_threat=p_thr,
                    sfi_score=sfi,
                    naive_cosine_sys=n_sys,
                    naive_cosine_thr=n_thr,
                    is_blocked=is_blocked,
                    latency_ms=lat,
                )
                self.telemetry_logs.append(log_entry)

                status_str = "🛑 [BLOCKED]" if is_blocked else "✅ [PASSED]"
                print(f"  [{status_str}] SFI: {sfi:+.4f} | Latency: {lat:.2f}ms | Text: {text[:60]}...")
