"""
E15_defense_aware_e12.py
================================================================================
NEED.md task 1: transfer the E6c defense-aware adaptive adversary protocol to
the E12 4th-block data (AdvBench + Alpaca-safe, HarmBench + Alpaca-safe).

E6c established the defense-aware adaptive attack protocol on the three
style-diverse datasets (Wild / ToxicChat / XSTest). This script replicates it
VERBATIM - same attack engine, same synonym banks, same detectors, same
calibration, same scenario set - on the behavior-level benchmark datasets of
E12, where clean ROC-AUC of the discriminant family is near-ceiling
(B1 ~ 0.997-1.0 on HarmBench, data/results/E12_advbench_harmbench_e2d.csv).

Reused via import (no method re-implemented, mirroring how E12 reuses E2d/E8):
  - from E6c_defense_aware_adaptive_attack.py:
      DefenseAwareAttacker, ExternalClassifierWrapper,
      compute_all_detector_scores, compute_calibration_thresholds,
      evaluate_adaptive_seed, print_summary
  - from E12_advbench_harmbench_extension.py:
      build_behavior_datasets, get_embeddings_any

Documented design choices (differences from E6c):
  1. Reference budget is 200/200 per seed (E12 rule; both classes >= 250),
     attack subsample is 60 malicious prompts per (dataset, embedder, seed)
     - identical to E6c.
  2. Scenarios (identical to E6c):
       adaptive_word_greedy_target_B1, adaptive_affix_target_B1,
       adaptive_combined_target_B1, adaptive_word_greedy_target_B1w.
  3. Unlike E6c there is NO contrastive safe-only dataset here (XSTest was
     the contrastive regime in E6c). Therefore the summary also reports
     ASR_adv - ASR_clean (delta) per method: at near-ceiling clean AUC the
     calibrated thresholds themselves can produce a non-zero clean ASR, and
     the delta isolates the attack effect from the operating-point effect.
     The per-seed clean rows are recorded for every method exactly as in E6c.
  4. Qwen/Qwen3-Embedding-8B is EXCLUDED by default: the E6c attack engine
     issues hundreds of incremental model.encode calls inside the greedy
     loop (score_batch on up to ~70 candidates per iteration), which is
     prohibitively slow for an 8B backbone on RTX 3060 (12 GB). Set
     E15_INCLUDE_QWEN=1 in the environment to opt in; the Qwen path uses a
     persistent manual encoder with the exact E2e fallback semantics
     (_encode_qwen_manual: last non-padding token hidden state, bf16,
     device_map=auto, MAX_LENGTH=2048) exposed through a
     SentenceTransformer-like .encode() API required by DefenseAwareAttacker.
  5. External transferability classifiers (deberta-v3-prompt-injection-v2,
     toxic-bert) are loaded exactly as in E6c; --no-externals skips them.

Outputs (data/results/):
  E15_defense_aware_e12.csv  (schema identical to E6c_defense_aware_adaptive_attack.csv)
"""

import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E6c_defense_aware_adaptive_attack import (  # noqa: E402
    DefenseAwareAttacker,
    ExternalClassifierWrapper,
    compute_all_detector_scores,
    compute_calibration_thresholds,
    evaluate_adaptive_seed,
    print_summary,
)
from E12_advbench_harmbench_extension import (  # noqa: E402
    build_behavior_datasets,
    get_embeddings_any,
)
from E2d_safe_aware_multidataset import DEVICE  # noqa: E402

ROOT = Path(__file__).parent.parent
OUT_DIR = ROOT / "data" / "results"
OUT_CSV = OUT_DIR / "E15_defense_aware_e12.csv"

STANDARD_EMBEDDERS = [
    "sentence-transformers/all-mpnet-base-v2",
    "BAAI/bge-base-en-v1.5",
    "BAAI/bge-large-en-v1.5",
]
QWEN_ID = "Qwen/Qwen3-Embedding-8B"


def include_qwen() -> bool:
    """Qwen3-8B participation is opt-in via env (see module docstring, item 4)."""
    return os.environ.get("E15_INCLUDE_QWEN", "0") == "1"


class ManualQwenEncoder:
    """Persistent Qwen3-Embedding-8B encoder with SentenceTransformer-like API.

    Embedding semantics are identical to E2e `_encode_qwen_manual`
    (last non-padding token hidden state, bf16, device_map=auto,
    MAX_LENGTH=2048, unnormalized float32 output) - the same definition that
    produced the committed `emb_cache/*_Qwen_Qwen3-Embedding-8B.npy` caches.
    The model is loaded ONCE and kept alive for the whole attack loop
    (the E6c engine needs hundreds of incremental .encode calls).
    """

    MAX_LENGTH = 2048

    def __init__(self):
        from transformers import AutoModel, AutoTokenizer

        self.tokenizer = AutoTokenizer.from_pretrained(QWEN_ID)
        self.model = AutoModel.from_pretrained(
            QWEN_ID,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            max_memory={0: "10GB", "cpu": "20GB"} if DEVICE == "cuda" else None,
        )
        self.model.eval()
        print(f"  [ManualQwenEncoder] loaded, dtype={next(self.model.parameters()).dtype}",
              flush=True)

    def encode(self, texts, convert_to_numpy: bool = True,
               show_progress_bar: bool = False, batch_size: int = 16) -> np.ndarray:
        # The E6c engine passes batch_size=128 for the 768d sentence encoders;
        # cap it for the 8B backbone to stay within a 12 GB GPU.
        batch_size = min(int(batch_size), 16)
        out = []
        with torch.no_grad():
            for i in range(0, len(texts), batch_size):
                batch = texts[i:i + batch_size]
                enc = self.tokenizer(
                    batch, padding=True, truncation=True,
                    max_length=self.MAX_LENGTH, return_tensors="pt",
                )
                if DEVICE == "cuda":
                    enc = {k: v.to("cuda") for k, v in enc.items()}
                hs = self.model(**enc).last_hidden_state          # (B, L, D)
                last_idx = (enc["attention_mask"].sum(dim=1) - 1).to(hs.device)
                vecs = hs[torch.arange(hs.size(0), device=hs.device), last_idx]
                out.append(vecs.detach().float().cpu().numpy())
        return np.vstack(out)


def make_encoder(model_id: str):
    """SentenceTransformer for the standard embedders, ManualQwenEncoder for Qwen."""
    if "Qwen3" in model_id:
        return ManualQwenEncoder()
    from sentence_transformers import SentenceTransformer

    return SentenceTransformer(model_id, device=DEVICE)


def load_externals() -> list:
    """External neural classifiers for the transferability audit (as in E6c)."""
    print("\nLoading External Neural Classifiers for Transferability Audit...")
    try:
        deberta = ExternalClassifierWrapper(
            "protectai/deberta-v3-base-prompt-injection-v2",
            score_mode="softmax_class1",
        )
        toxicbert = ExternalClassifierWrapper("unitary/toxic-bert", score_mode="sigmoid_max")
        return [
            ("deberta-v3-prompt-injection-v2", deberta),
            ("toxic-bert", toxicbert),
        ]
    except Exception as e:
        print(f"  Warning: Could not load external classifiers: {e}")
        return []


def print_delta_summary(df):
    """ASR_adv - ASR_clean per (dataset, model, evaluated_method, scenario).

    Rationale (module docstring, item 3): with no contrastive safe-only
    dataset, the calibrated thresholds can yield a non-zero clean ASR at
    near-ceiling AUC; the delta isolates the attack effect.
    """
    import pandas as pd

    print("\n" + "=" * 80)
    print("SUMMARY: ASR@1%FPR delta (adaptive attack MINUS clean) in %")
    print("=" * 80)
    clean = (
        df[df.attack_scenario == "clean"]
        .groupby(["dataset", "model", "evaluated_method"])["asr_at_1pct_fpr"]
        .mean()
        .rename("asr_clean")
    )
    adv = df[df.attack_scenario != "clean"].copy()
    adv["asr_clean_ref"] = [
        clean.loc[(r.dataset, r.model, r.evaluated_method)]
        for r in adv.itertuples()
    ]
    adv["asr_delta"] = adv["asr_at_1pct_fpr"] - adv["asr_clean_ref"]
    piv = adv.pivot_table(
        index=["dataset", "model", "evaluated_method"],
        columns="attack_scenario",
        values="asr_delta",
        aggfunc="mean",
    )
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print((piv * 100).round(2).to_string())


def run_benchmark(smoke: bool = False, include_externals: bool = True):
    """Run the E15 defense-aware suite on the E12 datasets."""
    import pandas as pd

    print("=" * 80)
    print("E15: DEFENSE-AWARE ADAPTIVE ATTACKS ON E12 DATA (AdvBench / HarmBench)")
    print("Protocol replicated verbatim from E6c; methods/datasets reused via import")
    print("=" * 80, flush=True)

    t_start = time.time()

    datasets = build_behavior_datasets()
    if smoke:
        datasets = {"AdvBench": datasets["AdvBench"]}

    embedders = list(STANDARD_EMBEDDERS)
    if include_qwen():
        embedders.append(QWEN_ID)
    if smoke:
        embedders = [STANDARD_EMBEDDERS[0]]

    externals = []
    if include_externals and not smoke:
        externals = load_externals()

    n_seeds = 1 if smoke else 5
    max_attacks = 25 if smoke else 60

    all_rows = []
    for d_name, (texts, labels) in datasets.items():
        labels = np.array(labels)
        n_mal = int(labels.sum())
        n_safe = len(labels) - n_mal
        print(f"\n{'#' * 80}\nDATASET: {d_name} ({len(texts)} samples: "
              f"{n_mal} mal / {n_safe} safe)\n{'#' * 80}", flush=True)

        # Precompute external scores on the clean full dataset ONCE (as in E6c)
        ext_base_scores = {}
        if externals:
            print(f"Pre-scoring clean {d_name} with external models...")
            for ext_name, ext_clf in externals:
                ext_base_scores[ext_name] = ext_clf.score_texts(texts)
            print("  Pre-scoring complete!")

        n_ref_mal = n_ref_safe = 200  # E12 budget rule (both classes >= 250)

        for model_id in embedders:
            model_short = model_id.split("/")[-1]
            print(f"\n--- Embedder: {model_short} ---", flush=True)

            # Load model ONCE per embedder loop
            encoder = make_encoder(model_id)
            embeddings = get_embeddings_any(texts, d_name, model_id)

            for seed in range(n_seeds):
                seed_rows = evaluate_adaptive_seed(
                    model=encoder,
                    model_name=model_id,
                    dataset_name=d_name,
                    texts=texts,
                    labels=labels,
                    embeddings=embeddings,
                    seed=seed,
                    n_ref_mal=n_ref_mal,
                    n_ref_safe=n_ref_safe,
                    max_attack_samples=max_attacks,
                    externals=externals,
                    ext_base_scores=ext_base_scores,
                )
                all_rows.extend(seed_rows)

    df = pd.DataFrame(all_rows)
    OUT_DIR.mkdir(parents=True, exist_ok=True)
    df.to_csv(OUT_CSV, index=False)
    print(f"\nSaved: {OUT_CSV} ({len(df)} rows, {time.time() - t_start:.1f}s)", flush=True)

    print_summary(df)
    print_delta_summary(df)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E15 defense-aware adaptive attack on E12 data")
    parser.add_argument("--smoke", action="store_true", help="Run fast smoke test")
    parser.add_argument("--no-externals", action="store_true", help="Skip external neural models")
    args = parser.parse_args()

    run_benchmark(smoke=args.smoke, include_externals=not args.no_externals)
