"""
verify_prop1.py
===============
Independent numerical verifier for Proposition 1 / Theorem 1 / Corollary 1
(docs/PROPOSITION_1.md).

Reads committed embedding caches (emb_cache/*.npy) and the committed
E2d_safe_aware_multidataset pipeline (seeds/splits). The verifier only checks
the algebraic identities on the SAME per-seed reference pools that produced
the committed CSVs — no method re-implementation beyond the definitions.

Checks:
  P1.i  : || T(mu_m) - T(mu_s) - W d || / ||W d||  ~= 0   (Proposition 1.i)
  P1.ii : max_x | B1_Sigma(x) - smc_{Sigma^{-1}}(x; d) |  = 0 per point
  T1    : || tr(Sigma_T - Sigma_W - pi_m*pi_s*d d^T) || / tr(Sigma_T) ~= 0
  C1    : | ||d||_{Sigma_T^{-1}} - ||d||_{Sigma_W^{-1}}/sqrt(1+pi_m pi_s rW^2) |
          / ||d||_{Sigma_T^{-1}}  ~= 0                               (Corollary 1)

For each (dataset, seed) compute identities -> scratch/prop1_verification.csv
"""

import sys, csv
from pathlib import Path

import numpy as np

ROOT = Path(__file__).parent.parent
sys.path.insert(0, str(ROOT / "src"))
sys.path.insert(0, str(ROOT / "experiments"))

from E2d_safe_aware_multidataset import (  # noqa: E402
    load_toxicchat, load_wild, load_xstest, get_embeddings,
    score_discriminant_mean,
)
from rsfi.whitening import SphericalWhitening  # noqa: E402

N_SEEDS = 5
N_REF = 200
OUT = Path(__file__).parent / "prop1_verification.csv"


def smc_M(x, mu, a_wh, W):
    """Cosine of Definition 2 with a NORMALIZED whitened anchor a_wh=Wd/||Wd||:
    B1b code projects onto u = Wd/||Wd||, so the reference is already in the
    whitened space. Returns (x-mu)^T W^T (Wd/||Wd||) / (||W(x-mu)||) — exactly
    the code path."""
    z = x - mu
    Wz = W @ z
    num = Wz @ a_wh
    den = np.linalg.norm(Wz)
    return num / den


def run_ds(name, texts, labels):
    mal_idx = np.where(np.array(labels) == 1)[0]
    safe_idx = np.where(np.array(labels) == 0)[0]
    rows = []
    for seed in range(N_SEEDS):
        rng = np.random.RandomState(seed)
        rm = rng.choice(mal_idx, N_REF, replace=False)
        rs = rng.choice(safe_idx, N_REF, replace=False)
        # SAME model as committed E2d config: Wild/ToxicChat -> bge-base,
        # XSTest -> mpnet. For Wild, cache is the shared 2000-row file which
        # covers only the first 2000 Wild records.
        if name == "XSTest":
            model = "sentence-transformers/all-mpnet-base-v2"
        else:
            model = "BAAI/bge-base-en-v1.5"
        emb = get_embeddings(texts, name, model)
        if emb.shape[0] < len(texts):
            # Wild: use only the cached prefix (matches E2d committed protocol)
            emb = emb[: len(texts)]
        Xm = emb[rm]; Xs = emb[rs]
        Xc = np.vstack([Xm, Xs])
        d = Xm.mean(axis=0) - Xs.mean(axis=0)
        nm, ns = len(rm), len(rs)
        pim, pis = nm / (nm + ns), ns / (nm + ns)

        # raw covariances (exact, no shrinkage -> T1 identity exact)
        mu_m, mu_s = Xm.mean(0), Xs.mean(0)
        mu_c = pim * mu_m + pis * mu_s
        Sm = ((Xm - mu_m).T @ (Xm - mu_m)) / nm
        Ss = ((Xs - mu_s).T @ (Xs - mu_s)) / ns
        Sc_raw = ((Xc - mu_c).T @ (Xc - mu_c)) / (nm + ns)
        Sw_raw = pim * Sm + pis * Ss
        tr_T = np.trace(Sc_raw)
        T1_rel = abs(np.trace(Sc_raw - Sw_raw - pim * pis * np.outer(d, d))) / tr_T

        # Sigma_T whitening as in E2d (Ledoit-Wolf on combined)
        wh = SphericalWhitening(dim=emb.shape[1]); wh.fit(Xc)
        W = wh.W                                    # symmetric Sigma^{-1/2}
        mu_wh = wh.mu
        Tm = W @ (mu_m - mu_wh)
        Ts = W @ (mu_s - mu_wh)
        Wd = W @ d
        P1_i_abs = np.linalg.norm((Tm - Ts) - Wd)
        P1_i_rel = P1_i_abs / (np.linalg.norm(Wd) + 1e-300)

        Minv_T = W @ W                              # = Sigma_T^{-1}
        test_idx = np.concatenate([np.setdiff1d(mal_idx, rm),
                                   np.setdiff1d(safe_idx, rs)])
        rngt = np.random.RandomState(0)
        test_pts = rngt.choice(test_idx, min(200, len(test_idx)), replace=False)
        Xt = emb[test_pts]

        wh_test = wh.transform(Xt)
        wh_m = wh.transform(Xm)
        wh_s = wh.transform(Xs)
        B1b_wh = score_discriminant_mean(wh_test, wh_m, wh_s)
        # Proposition 1.ii: B1b must equal smc with NORMALIZED whitened anchor.
        u_norm = Wd / (np.linalg.norm(Wd) + 1e-300)
        smc_vals = np.array([smc_M(x, mu_wh, u_norm, W) for x in Xt])
        P1_ii_max = float(np.max(np.abs(B1b_wh - smc_vals)))

        # Corollary 1: Sherman-Morrison closed form vs direct.
        # Use the SAME shrunk covariance estimates as the E8 code path:
        # Sigma_W = LedoitWolf on each class (centered at own mean), pooled;
        # Sigma_T = LedoitWolf on combined pool (centered at mu_c).
        from sklearn.covariance import LedoitWolf
        lw_m = LedoitWolf().fit(Xm - mu_m)
        lw_s = LedoitWolf().fit(Xs - mu_s)
        Sw = 0.5 * (lw_m.covariance_ + lw_s.covariance_)
        lw_t = LedoitWolf().fit(Xc - mu_c)
        St_inv = np.linalg.inv(lw_t.covariance_)
        evals_w, evecs_w = np.linalg.eigh(Sw)
        evals_w = np.maximum(evals_w, 1e-12)
        Sw_inv = evecs_w @ np.diag(1.0 / evals_w) @ evecs_w.T
        rW = float(np.sqrt(max(d @ Sw_inv @ d, 0.0)))
        dT_norm = float(np.sqrt(max(d @ St_inv @ d, 0.0)))
        pred = rW / np.sqrt(1 + pim * pis * rW * rW)
        C1_rel = abs(dT_norm - pred) / (dT_norm + 1e-300)

        rows.append({
            "dataset": name, "seed": seed, "dim": emb.shape[1],
            "n_ref_mal": nm, "n_ref_safe": ns,
            "P1_i_abs_err": f"{P1_i_abs:.3e}", "P1_i_rel": f"{P1_i_rel:.3e}",
            "P1_ii_max_abs": f"{P1_ii_max:.3e}", "T1_rel_trace": f"{T1_rel:.3e}",
            "C1_rel_err": f"{C1_rel:.3e}", "r_W": f"{rW:.4f}",
        })
        print(f"[{name} seed {seed}] P1.i rel={P1_i_rel:.2e}  "
              f"P1.ii max={P1_ii_max:.2e}  T1 rel={T1_rel:.2e}  "
              f"C1 rel={C1_rel:.2e}  r_W={rW:.3f}")
    return rows


def main():
    datasets = [
        ("Wild", *load_wild()),
        ("ToxicChat", *load_toxicchat()),
        ("XSTest", *load_xstest()),
    ]
    rows = []
    for name, texts, labels in datasets:
        print(f"\n=== {name} ===")
        rows.extend(run_ds(name, texts, labels))

    with open(OUT, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0].keys()))
        w.writeheader()
        w.writerows(rows)
    print(f"\nSaved {len(rows)} rows -> {OUT}")


if __name__ == "__main__":
    main()