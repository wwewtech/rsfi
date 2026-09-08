import os
from pathlib import Path

# Ensure writable local cache directory for HuggingFace models
CACHE_DIR = os.path.abspath("./hf_cache")
os.makedirs(CACHE_DIR, exist_ok=True)
os.environ["HF_HOME"] = CACHE_DIR
os.environ["SENTENCE_TRANSFORMERS_HOME"] = CACHE_DIR

import argparse
from dataclasses import asdict
import json
import sys

# Ensure src is in python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from rsfi import WildChatBenchmarkRunner


def main():
    parser = argparse.ArgumentParser(
        description="RSFI Benchmark Runner for WildChat 10k Datasets."
    )
    parser.add_argument(
        "--samples-per-class",
        type=int,
        default=1000,
        help="Number of samples per class (default: 1000, total: 2000)",
    )
    parser.add_argument(
        "--model-name",
        type=str,
        default="sentence-transformers/paraphrase-multilingual-MiniLM-L12-v2",
        help="Embedding model name",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default="data/reports",
        help="Directory to save CSV and JSON reports",
    )
    parser.add_argument(
        "--tau", type=float, default=-0.15, help="RSFI decision threshold tau"
    )
    args = parser.parse_args()

    output_path = Path(args.output_dir)
    output_path.mkdir(parents=True, exist_ok=True)

    print("=" * 85)
    print(
        "   RUNNING RSFI BENCHMARK ON REAL-WORLD PROMPT DATASETS (WILDCHAT / IN-THE-WILD)"
    )
    print("=" * 85 + "\n")

    runner = WildChatBenchmarkRunner(model_name=args.model_name, cache_folder=CACHE_DIR)
    samples = runner.load_dataset_samples(target_per_class=args.samples_per_class)

    print(
        f"\n[EVALUATING] Processing {len(samples)} real-world prompts with threshold tau = {args.tau}..."
    )
    report, df_telemetry = runner.run_benchmark(samples=samples, tau=args.tau)

    # Save CSV telemetry
    csv_file = output_path / "wildchat_10k_telemetry.csv"
    df_telemetry.to_csv(csv_file, index=False)

    # Save JSON summary report
    json_file = output_path / "wildchat_10k_summary.json"
    with open(json_file, "w", encoding="utf-8") as f:
        json.dump(asdict(report), f, indent=2)

    print("\n" + "=" * 85)
    print("     FINAL STATISTICAL REPORT ON IN-THE-WILD PROMPTS")
    print("=" * 85)
    print(f"Total evaluated prompts (N)            : {report.total_samples}")
    print(
        f"True Positives (Jailbreaks blocked)    : {report.true_positives} / {report.samples_per_class}"
    )
    print(
        f"True Negatives (Regular passed)        : {report.true_negatives} / {report.samples_per_class}"
    )
    print(
        f"False Positives (False bans)           : {report.false_positives} / {report.samples_per_class} (FPR: {report.false_positive_rate * 100:.2f}%)"
    )
    print(
        f"False Negatives (Missed attacks)       : {report.false_negatives} / {report.samples_per_class}"
    )
    print("-" * 85)
    print(
        f"Accuracy                               : {report.accuracy:.4f} ({report.accuracy * 100:.2f}%)"
    )
    print(f"Precision                              : {report.precision:.4f}")
    print(
        f"Recall (Mitigation Rate)              : {report.recall:.4f} ({report.recall * 100:.2f}%)"
    )
    print(f"F1-Score                               : {report.f1_score:.4f}")
    print(f"ROC-AUC Score                          : {report.roc_auc:.4f}")
    print(f"Mean Latency per evaluation            : {report.mean_latency_ms:.3f} ms")
    print(f"P95 Latency per evaluation             : {report.p95_latency_ms:.3f} ms")
    print("=" * 85)
    print(f"\n[EXPORT] CSV telemetry saved to: {csv_file}")
    print(f"[EXPORT] JSON summary report saved to: {json_file}\n")


if __name__ == "__main__":
    main()
