"""
E16_cross_domain_bigbench.py
================================================================================
NEED.md task 2: "Cross-domain benchmark expansion - add more datasets
(e.g., BigBench)."

What this adds: safe-pool datasets from the OFFICIAL google/BIG-bench repo
(raw.githubusercontent.com/bigbench/benchmark_tasks/<task>/task.json), a
source completely disjoint from the malicious content - the safe class
consists of formal/reasoning puzzle inputs, not instructions.

Dataset variants (malicious side identical to E12: AdvBench / HarmBench):
  *X_BB_Safe   : safe class = 1200 BigBench inputs (3 tasks x 400,
                 homogeneous "puzzle" domain)
  *X_BB_Mixed  : safe class = 200 Alpaca self-contained instructions
                 (disjoint slice of the E12 pool) + 400 BigBench inputs
                 (style-heterogeneous safe class - the discriminant must
                 handle a bimodal safe distribution)

Methods: the FULL E2d battery + E8 Sigma_W block, imported verbatim from
E12_advbench_harmbench_extension.run_battery (no method re-implemented).

BigBench tasks (selected with scratch/bigbench_probe.py, verified sizes on
2026-09-16 from the official repo):
  hyperbaton            50000 examples (adjective ordering)
  navigate               1000 examples (spatial navigation chains)
  temporal_sequences     1000 examples (schedule reasoning)
The downloader caches each task.json-derived input list to
data/raw/bigbench/<task>.json so re-runs are network-free.

Outputs (data/results/):
  E16_cross_domain_battery.csv
  E16_cross_domain_delong.csv
  E16_cross_domain_diagnostics.csv
"""

import json
import sys
import urllib.request
from pathlib import Path

import numpy as np
import pandas as pd

sys.path.insert(0, str(Path(__file__).parent))
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from E12_advbench_harmbench_extension import (  # noqa: E402
    ROOT, RAW, EMB_CACHE, EMBEDDERS, DEVICE, N_SEEDS,
    ALPACA_SEED, SAFE_POOL_ADV, SAFE_POOL_HB,
    get_embeddings_any, run_battery,
)

BB_DIR = RAW / "bigbench"
BB_TASKS = ["hyperbaton", "navigate", "temporal_sequences"]
BB_PER_TASK = 400
BB_SAFE_POOL = BB_PER_TASK * len(BB_TASKS)          # 1200 (BB_Safe variant)
BB_MIXED_POOL = BB_PER_TASK                          # 400  (BB_Mixed variant)
ALPACA_MIXED = 200
BB_SEED = 2001
BIGBENCH_URL = ("https://raw.githubusercontent.com/google/BIG-bench/main/"
                "bigbench/benchmark_tasks/{task}/task.json")

def fetch_bigbench_inputs(task: str) -> list:
    """Download (or load cached) task inputs from the official BIG-bench repo."""
    cache = BB_DIR / f"{task}.json"
    if cache.exists():
        return json.load(open(cache, encoding="utf-8"))
    url = BIGBENCH_URL.format(task=task)
    with urllib.request.urlopen(url, timeout=60) as r:
        data = json.loads(r.read().decode("utf-8"))
    inputs = [str(ex.get("input", "")).strip() for ex in data.get("examples", [])]
    inputs = [t for t in inputs if t]
    BB_DIR.mkdir(parents=True, exist_ok=True)
    json.dump(inputs, open(cache, "w", encoding="utf-8"),
              ensure_ascii=False, indent=0)
    print(f"  [downloaded] {task}: {len(inputs)} inputs -> {cache.name}")
    return inputs


def build_cross_domain_datasets(alpaca_pool: list) -> dict:
    """Build {name: (texts, labels)} for the two cross-domain variants per dataset."""
    adv = pd.read_csv(RAW / "advbench_harmful_behaviors.csv")
    hb = pd.read_csv(RAW / "harmbench_all.csv")
    adv_mal = adv["goal"].astype(str).tolist()
    hb_mal = hb["Behavior"].astype(str).tolist()

    bb = {t: fetch_bigbench_inputs(t) for t in BB_TASKS}
    rng = np.random.RandomState(BB_SEED)
    bb_sel = {}
    for t, inputs in bb.items():
        if len(inputs) < BB_PER_TASK:
            raise RuntimeError(
                f"BigBench task {t} has only {len(inputs)} inputs "
                f"(< {BB_PER_TASK} required)")
        idx = rng.choice(len(inputs), BB_PER_TASK, replace=False)
        bb_sel[t] = [inputs[i] for i in idx]

    # Disjoint Alpaca slice: replicate E12's shuffled pool (same ALPACA_SEED)
    # and take from the tail beyond E12's SAFE_POOL_ADV + SAFE_POOL_HB usage.
    used = SAFE_POOL_ADV + SAFE_POOL_HB
    order = np.random.RandomState(ALPACA_SEED).permutation(len(alpaca_pool))
    alpaca_slice = [alpaca_pool[i] for i in order[used:used + ALPACA_MIXED]]
    if len(alpaca_slice) < ALPACA_MIXED:
        raise RuntimeError(f"Alpaca pool too small: {len(alpaca_slice)}")

    def bb_texts(per_task=BB_PER_TASK):
        return sum(([bb_sel[t][i] for i in range(per_task)] for t in BB_TASKS), [])

    datasets = {}
    for src_name, mal in [("AdvBench", adv_mal), ("HarmBench", hb_mal)]:
        datasets[f"{src_name}_BB_Safe"] = (
            mal + bb_texts(), [1] * len(mal) + [0] * (BB_SAFE_POOL))
        datasets[f"{src_name}_BB_Mixed"] = (
            mal + alpaca_slice + bb_texts(BB_MIXED_POOL),
            [1] * len(mal) + [0] * ALPACA_MIXED + [0] * BB_MIXED_POOL)
    return datasets


def main(smoke: bool = False):
    from E12_advbench_harmbench_extension import load_alpaca_selfcontained

    print("=" * 80)
    print("E16: CROSS-DOMAIN EXPANSION - BIGBENCH SAFE POOLS (E12 battery)")
    print(f"BigBench tasks: {BB_TASKS} (official google/BIG-bench repo)")
    print("=" * 80, flush=True)

    datasets = build_cross_domain_datasets(load_alpaca_selfcontained())

    res_all, dlong_all, diag_all = [], [], []
    n_seeds = 1 if smoke else N_SEEDS
    for d_name, (texts, labels) in datasets.items():
        labels_arr = np.array(labels)
        mal_idx = np.where(labels_arr == 1)[0]
        safe_idx = np.where(labels_arr == 0)[0]
        n_ref = 200 if smoke else 200  # E12 budget rule
        print(f"\n### {d_name}: {len(texts)} items "
              f"({len(mal_idx)} mal / {len(safe_idx)} safe)", flush=True)

        for model_id in EMBEDDERS[:1] if smoke else EMBEDDERS:
            model_short = model_id.split("/")[-1]
            print(f"--- {d_name} x {model_short} ---", flush=True)
            embeddings = get_embeddings_any(texts, d_name, model_id)
            for seed in range(n_seeds):
                r, d, g = run_battery(
                    d_name, embeddings, labels_arr, seed, n_ref, n_ref,
                    mal_idx, safe_idx)
                for row in r:
                    row["model"] = model_short
                for row in d:
                    row["model"] = model_short
                for row in g:
                    row["model"] = model_short
                res_all.extend(r)
                dlong_all.extend(d)
                diag_all.extend(g)
                b1 = next(x["roc_auc"] for x in r
                          if x["method"] == "B1_discriminant_mean_raw")
                a1 = next(x["roc_auc"] for x in r
                          if x["method"] == "A1_naive_cosine_raw")
                print(f"  seed {seed}: B1={b1:.4f} A1={a1:.4f}", flush=True)

    out = ROOT / "data" / "results"
    out.mkdir(parents=True, exist_ok=True)
    pd.DataFrame(res_all).to_csv(out / "E16_cross_domain_battery.csv", index=False)
    pd.DataFrame(dlong_all).to_csv(out / "E16_cross_domain_delong.csv", index=False)
    pd.DataFrame(diag_all).to_csv(out / "E16_cross_domain_diagnostics.csv", index=False)

    df = pd.DataFrame(res_all)
    print("\n=== E16 ROC-AUC (mean over seeds) ===")
    piv = df.pivot_table(index=["dataset", "model"], columns="method",
                         values="roc_auc")
    with pd.option_context("display.max_columns", None, "display.width", 220):
        print(piv.round(4).to_string())
    print("\nSaved 3 CSV files to", out)


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="E16 cross-domain BigBench expansion")
    parser.add_argument("--smoke", action="store_true", help="1 seed, 1 embedder")
    args = parser.parse_args()

    main(smoke=args.smoke)
