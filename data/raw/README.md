# Raw data provenance

## AdvBench (Zou et al., 2023, arXiv:2307.15043)

- `advbench_harmful_behaviors.csv` — canonical 520 harmful behaviors,
  downloaded from https://github.com/llm-attacks/llm-attacks (data/advbench/).
  Verified: 520 rows, columns `goal`, `target`.
- The companion `harmless_behaviors.csv` (100 harmless behaviors) is NOT
  distributed in the llm-attacks repository (only via a Google Drive link);
  third-party mirrors mix corpora from different projects or add chat wrappers.
  Decision: AdvBench used as **malicious-only pool** (no harmless subset).

## HarmBench (Mazeika et al., 2024, arXiv:2402.05793)

- `harmbench_all.csv` — full official text behaviors release (400 behaviors:
  functional + contextual), downloaded from
  https://github.com/centerforaisafety/HarmBench
  (data/behavior_datasets/harmbench_behaviors_text_all.csv).
  Used in E12 as the heterogeneous malicious pool.

## Stanford Alpaca (Taori et al., 2023)

- `alpaca_data.json` — canonical 52k instruction-following dataset,
  downloaded from https://raw.githubusercontent.com/tatsu-lab/stanford_alpaca/
  main/alpaca_data.json. E12 selects the first 800 SELF-CONTAINED
  (no `input`) instructions by a fixed seed permutation to form disjoint
  400-safe pools for the AdvBench and HarmBench benchmarks.
  Style-homogeneous negative pool (imperative instructions vs. imperative
  malicious behaviors); the original AdvBench/HarmBench authors do not
  publish a harmless class.
