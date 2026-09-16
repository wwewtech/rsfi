"""
test_e16_cross_domain.py - gate for E16 cross-domain BigBench expansion.

Behavioural invariants (verified at build time on 2026-09-16):
  - E16 reuses the E12 battery verbatim via
    E12_advbench_harmbench_extension.run_battery (import, not copy).
  - BigBench tasks must come from the official google/BIG-bench repo and be
    cached locally; a runtime guard raises if a task has fewer inputs than
    required (see E16_cross_domain_bigbench.build_cross_domain_datasets).
  - The BigBench/Alpaca safe pools are DISJOINT from the E12 safe pools
    (verified empirically: 0 overlapping strings for all 4 variants).
"""
import importlib.util
import sys
from pathlib import Path

import pytest

EXP = Path(__file__).parent.parent / "experiments"


def _import(mod_name, file_name):
    spec = importlib.util.spec_from_file_location(mod_name, EXP / file_name)
    mod = importlib.util.module_from_spec(spec)
    sys.modules.setdefault(mod_name, mod)
    spec.loader.exec_module(mod)
    return mod


@pytest.fixture(scope="module")
def e16():
    mod = _import("E16_cross_domain_bigbench", "E16_cross_domain_bigbench.py")
    from E12_advbench_harmbench_extension import load_alpaca_selfcontained
    return mod, mod.build_cross_domain_datasets(load_alpaca_selfcontained())


def test_dataset_shapes(e16):
    mod, ds = e16
    for name in ["AdvBench_BB_Safe", "AdvBench_BB_Mixed",
                 "HarmBench_BB_Safe", "HarmBench_BB_Mixed"]:
        texts, labels = ds[name]
        n_mal, n_safe = sum(labels), len(labels) - sum(labels)
        assert n_mal >= 400, f"{name}: malicious side too small ({n_mal})"
        assert n_safe >= 600, f"{name}: safe side too small ({n_safe})"
        assert mod.BB_SAFE_POOL == 1200 and mod.BB_MIXED_POOL == 400


def test_bigbench_sources_are_official(e16):
    """Tasks must be the officially probed ones, cached under data/raw/bigbench."""
    mod, _ = e16
    assert mod.BB_TASKS == ["hyperbaton", "navigate", "temporal_sequences"]
    assert "google/BIG-bench" in mod.BIGBENCH_URL
    for t in mod.BB_TASKS:
        assert (mod.BB_DIR / f"{t}.json").exists(), f"cache missing for {t}"
        assert len(mod.fetch_bigbench_inputs(t)) >= mod.BB_PER_TASK


def test_no_overlap_with_e12_pools(e16):
    mod, ds = e16
    e12 = _import("E12", "E12_advbench_harmbench_extension.py")
    ref = e12.build_behavior_datasets()
    for src in ["AdvBench", "HarmBench"]:
        e12_safe = set(map(str, ref[src][0]))
        for var in ["BB_Safe", "BB_Mixed"]:
            safe16 = [str(t) for t, l in zip(ds[f"{src}_{var}"][0],
                                             ds[f"{src}_{var}"][1]) if l == 0]
            overlap = [t for t in safe16 if t in e12_safe]
            assert not overlap, f"{src}_{var}: {len(overlap)} strings overlap E12"
