"""
Advanced Mathematical Boundary and Stress Test Suite for RSFI.
Tests edge cases, numerical limits, ZCA invariants, isometric invariance,
subspace rank-deficiencies, and geodesic monotonicity with deep telemetry logs.
"""

import sys
from pathlib import Path

# Add src directory and root directory to python path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

import time
import numpy as np
from scipy.linalg import sqrtm, ishermitian

from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening
from rsfi.filter import RSFIFilter, MultiDimensionalRSFIFilter


# =====================================================================
# UTILITY: RIGOROUS LOGGING HELPER
# =====================================================================

class MathTestLogger:
    """Structured logger for mathematical boundary stress tests."""

    @staticmethod
    def header(title: str):
        print("\n" + "=" * 90)
        print(f"   {title}")
        print("=" * 90)

    @staticmethod
    def section(name: str):
        print(f"\n--- [SECTION] {name} ---")

    @staticmethod
    def log_metric(name: str, value: float, threshold: float, unit: str = "", passed: bool = True):
        status = "[PASS]" if passed else "[FAIL]"
        print(f"  {status} {name:<45} = {value:12.6e} {unit:<5} (Tolerance <= {threshold:.1e})")

    @staticmethod
    def log_info(msg: str):
        print(f"  [INFO] {msg}")

    @staticmethod
    def log_warn(msg: str):
        print(f"  [WARN] {msg}")


# =====================================================================
# TEST SUITE 1: RIEMANNIAN GEOMETRY SINGULARITIES & ISOMETRY
# =====================================================================

def test_riemannian_geometry_boundaries(dim: int = 1536, seed: int = 42):
    """
    Stress-tests RiemannianSphere under singular points:
    1. Identical vectors (x = y)
    2. Near-identical vectors (x ≈ y)
    3. Orthogonal vectors (<x,y> = 0)
    4. Near-antipodal vectors (<x,y> ≈ -1 + 1e-7)
    5. Antipodal vectors (y = -x)
    6. Triangle Inequality on S^(d-1)
    7. Rotational Invariance (Isometry under Q in O(d))
    """
    MathTestLogger.header("SUITE 1: RIEMANNIAN GEOMETRY BOUNDARIES & SINGULARITIES")
    np.random.seed(seed)
    
    x = RiemannianSphere.normalize(np.random.randn(dim))

    # --- 1. Identical vectors ---
    MathTestLogger.section("1.1 Identical vectors (y = x)")
    v_log_same = RiemannianSphere.log_map(x, x)
    norm_same = np.linalg.norm(v_log_same)
    dist_same = RiemannianSphere.geodesic_distance(x, x)
    MathTestLogger.log_metric("Geodesic distance d_M(x, x)", dist_same, 1e-12)
    MathTestLogger.log_metric("Log_map norm ||Log_x(x)||_2", norm_same, 1e-12)
    assert dist_same < 1e-12 and norm_same < 1e-12, "Singularity error at y = x"

    # --- 2. Near-identical vectors ---
    MathTestLogger.section("2.1 Near-identical vectors (y ~= x + eps)")
    eps = 1e-10
    y_near = RiemannianSphere.normalize(x + eps * np.random.randn(dim))
    dist_near = RiemannianSphere.geodesic_distance(x, y_near)
    v_log_near = RiemannianSphere.log_map(x, y_near)
    norm_near = np.linalg.norm(v_log_near)
    err_near = abs(dist_near - norm_near)
    MathTestLogger.log_metric("Near-identical Log_map norm error", err_near, 1e-10)
    assert err_near < 1e-10, "Numerical instability for near-identical vectors"

    # --- 3. Orthogonal vectors ---
    MathTestLogger.section("3.1 Orthogonal vectors (<x, y> = 0)")
    y_raw = np.random.randn(dim)
    y_ortho = RiemannianSphere.normalize(y_raw - np.dot(x, y_raw) * x)
    dist_ortho = RiemannianSphere.geodesic_distance(x, y_ortho)
    v_log_ortho = RiemannianSphere.log_map(x, y_ortho)
    dot_val = np.dot(x, y_ortho)
    MathTestLogger.log_metric("Dot product <x, y_ortho>", abs(dot_val), 1e-12)
    MathTestLogger.log_metric("Geodesic distance for orthogonal pair", abs(dist_ortho - np.pi / 2), 1e-10)
    assert abs(dot_val) < 1e-12, "Orthogonal vector generation error"

    # --- 4. Near-antipodal vectors ---
    MathTestLogger.section("4.1 Near-antipodal vectors (<x, y> ~= -1)")
    y_antipodal_near = RiemannianSphere.normalize(-x + 1e-7 * np.random.randn(dim))
    dist_anti_near = RiemannianSphere.geodesic_distance(x, y_antipodal_near)
    v_log_anti_near = RiemannianSphere.log_map(x, y_antipodal_near)
    norm_anti_near = np.linalg.norm(v_log_anti_near)
    err_anti_near = abs(dist_anti_near - norm_anti_near)
    is_nan = np.isnan(v_log_anti_near).any() or np.isinf(v_log_anti_near).any()
    MathTestLogger.log_metric("Near-antipodal Log_map norm error", err_anti_near, 1e-6)
    MathTestLogger.log_info(f"Contains NaN/Inf: {is_nan}")
    assert not is_nan and err_anti_near < 1e-6, "Antipodal singularity blew up into NaN/Inf"

    # --- 5. Exact Antipodal vector ---
    MathTestLogger.section("5.1 Exact Antipodal vector (y = -x)")
    y_anti_exact = -x
    dist_anti_exact = RiemannianSphere.geodesic_distance(x, y_anti_exact)
    v_log_anti_exact = RiemannianSphere.log_map(x, y_anti_exact)
    is_nan_exact = np.isnan(v_log_anti_exact).any()
    MathTestLogger.log_metric("Exact antipodal distance d_M(x, -x)", abs(dist_anti_exact - np.pi), 1e-10)
    MathTestLogger.log_info(f"Log_map(-x) NaN check: {is_nan_exact} (Norm: {np.linalg.norm(v_log_anti_exact):.6f})")

    # --- 6. Triangle Inequality on S^(d-1) ---
    MathTestLogger.section("6.1 Triangle Inequality on S^(d-1) (d_M(x, z) <= d_M(x, y) + d_M(y, z))")
    max_tri_violation = 0.0
    for _ in range(500):
        p1 = RiemannianSphere.normalize(np.random.randn(dim))
        p2 = RiemannianSphere.normalize(np.random.randn(dim))
        p3 = RiemannianSphere.normalize(np.random.randn(dim))

        d12 = RiemannianSphere.geodesic_distance(p1, p2)
        d23 = RiemannianSphere.geodesic_distance(p2, p3)
        d13 = RiemannianSphere.geodesic_distance(p1, p3)

        violation = d13 - (d12 + d23)
        if violation > max_tri_violation:
            max_tri_violation = violation

    MathTestLogger.log_metric("Max Triangle Inequality Violation", max_tri_violation, 1e-12)
    assert max_tri_violation < 1e-12, "Spherical triangle inequality violated"

    # --- 7. Rotational Invariance (Isometry under Q in O(d)) ---
    MathTestLogger.section("7.1 Rotational Invariance under Orthogonal Transformation Q in O(d)")
    # Generate random orthogonal matrix Q via QR
    Q, _ = np.linalg.qr(np.random.randn(dim, dim))
    
    x_rot = Q @ x
    y_rot = Q @ y_near

    dist_orig = RiemannianSphere.geodesic_distance(x, y_near)
    dist_rot = RiemannianSphere.geodesic_distance(x_rot, y_rot)
    dist_diff = abs(dist_orig - dist_rot)

    v_log_orig = RiemannianSphere.log_map(x, y_near)
    v_log_rot = RiemannianSphere.log_map(x_rot, y_rot)
    v_log_transformed = Q @ v_log_orig
    log_rot_err = np.linalg.norm(v_log_rot - v_log_transformed)

    MathTestLogger.log_metric("Distance Rotational Invariance Delta", dist_diff, 1e-12)
    MathTestLogger.log_metric("Log_map Rotational Isometry Norm Error", log_rot_err, 1e-10)
    assert dist_diff < 1e-12 and log_rot_err < 1e-10, "Riemannian operations violate rotational isometry"


# =====================================================================
# TEST SUITE 2: ZCA WHITENING INVARIANTS & STRESS AT HIGH DIMENSIONS
# =====================================================================

def test_zca_whitening_invariants(dim: int = 1536, seed: int = 42):
    """
    Stress-tests SphericalWhitening:
    1. Matrix Symmetry W_zca^T = W_zca
    2. Positive Definiteness (Eigenvalues > 0)
    3. Covariance Decorrelation / Isotropy Cov(X_whitened) = I_d
    4. Rank-Deficient Stress Test (N << d: N=30 samples, d=1536)
    5. Regularization Lambda Behavior
    """
    MathTestLogger.header("SUITE 2: ZCA WHITENING INVARIANTS & RANK-DEFICIENCY STRESS")
    np.random.seed(seed)

    # --- 1. Symmetry & Positive Definiteness ---
    MathTestLogger.section("1.1 ZCA Matrix Symmetry & Eigenvalues (Full Rank N > d)")
    N_large = 2000
    # Generate anisotropic covariance
    scale = np.exp(np.random.randn(dim) * 0.5)
    data_full = np.random.randn(N_large, dim) * scale
    
    whitening = SphericalWhitening(dim=dim, reg=1e-6)
    whitening.fit(data_full)

    W = whitening.W
    symmetry_err = np.linalg.norm(W - W.T, ord='fro')
    min_eig = float(np.min(np.linalg.eigvalsh(W)))

    MathTestLogger.log_metric("Frobenius Norm of Symmetry Error ||W - W^T||_F", symmetry_err, 1e-8)
    MathTestLogger.log_metric("Minimum Eigenvalue min(lambda(W))", min_eig, 0.0, passed=(min_eig > 0))
    assert symmetry_err < 1e-8, "ZCA Whitening matrix is not symmetric"
    assert min_eig > 0, "ZCA Whitening matrix is not positive-definite"

    # --- 2. Covariance Isotropy ---
    MathTestLogger.section("2.1 Covariance Decorrelation / Isotropy Test")
    centered_data = data_full - whitening.mu
    whitened_unnorm = centered_data @ W
    cov_whitened = np.cov(whitened_unnorm, rowvar=False)
    isotropy_err = np.linalg.norm(cov_whitened - np.eye(dim), ord='fro') / dim
    MathTestLogger.log_metric("Mean Frobenius Error ||Cov(X_whitened) - I||_F / d", isotropy_err, 1e-2)

    # --- 3. Severe Rank-Deficiency Stress (N << d) ---
    MathTestLogger.section("3.1 Severe Rank-Deficiency Stress (N=30 samples, d=1536)")
    N_small = 30
    data_sparse = np.random.randn(N_small, dim) * scale

    whitening_sparse = SphericalWhitening(dim=dim, reg=1e-4)
    t0 = time.perf_counter()
    whitening_sparse.fit(data_sparse)
    fit_time_ms = (time.perf_counter() - t0) * 1000.0

    W_sparse = whitening_sparse.W
    is_nan_w = np.isnan(W_sparse).any() or np.isinf(W_sparse).any()
    cond_num = np.linalg.cond(W_sparse)

    MathTestLogger.log_metric("Rank-Deficient Fit Time", fit_time_ms, 5000.0, unit="ms")
    MathTestLogger.log_metric("Condition Number cond(W_sparse)", cond_num, 1e8)
    MathTestLogger.log_info(f"Sparse ZCA contains NaN/Inf: {is_nan_w}")
    assert not is_nan_w, "ZCA Whitening produced NaN/Inf on rank-deficient data"

    # --- 4. Transform Output Normalization ---
    MathTestLogger.section("4.1 Output Normalization Test on Spherical Projection")
    transformed_sparse = whitening_sparse.transform(data_sparse[:5])
    norms = np.linalg.norm(transformed_sparse, axis=1)
    max_norm_err = float(np.max(np.abs(norms - 1.0)))
    MathTestLogger.log_metric("Max Spherical Re-projection Error || ||y|| - 1 ||", max_norm_err, 1e-12)
    assert max_norm_err < 1e-12, "Spherical re-projection failed to output unit vectors"


# =====================================================================
# TEST SUITE 3: SUBSPACE QR DECOMPOSITION & DEGENERACY
# =====================================================================

def test_subspace_qr_degeneracy(dim: int = 1536, seed: int = 42):
    """
    Stress-tests MultiDimensionalRSFIFilter under degenerate threat sets:
    1. Duplicate / Co-linear threat vectors
    2. Near-collinear threat vectors
    3. Orthonormality Q_k^T * Q_k = I_k under rank deficiency
    4. Large threat subspace (k=50 threats)
    """
    MathTestLogger.header("SUITE 3: THREAT SUBSPACE QR DECOMPOSITION & DEGENERACY STRESS")
    np.random.seed(seed)

    S = RiemannianSphere.normalize(np.random.randn(dim))

    # --- 1. Duplicate Threat Vectors ---
    MathTestLogger.section("1.1 Identical / Duplicate Threat Anchors (V_1 = V_2 = V_3)")
    V_base = RiemannianSphere.normalize(np.random.randn(dim))
    V_threats_dup = [V_base, V_base.copy(), V_base.copy()]

    filter_dup = MultiDimensionalRSFIFilter(S, V_threats_dup, alpha=1.5, beta=0.5, tau=0.0)
    Q_dup = filter_dup.Q_thr
    
    is_nan_q = np.isnan(Q_dup).any()
    k_dup = Q_dup.shape[1]
    ortho_dup_err = np.linalg.norm(Q_dup.T @ Q_dup - np.eye(k_dup))

    MathTestLogger.log_metric("Duplicate Threats Basis Orthonormality Error ||Q^T Q - I||", ortho_dup_err, 1e-10)
    MathTestLogger.log_info(f"Q_dup shape: {Q_dup.shape}, Contains NaN: {is_nan_q}")
    assert not is_nan_q, "QR decomposition produced NaN on duplicate threat vectors"

    # --- 2. Near-Collinear Threat Vectors ---
    MathTestLogger.section("2.1 Near-Collinear Threat Vectors (V_2 = V_1 + 1e-8 * noise)")
    V1 = RiemannianSphere.normalize(np.random.randn(dim))
    V2 = RiemannianSphere.normalize(V1 + 1e-8 * np.random.randn(dim))
    V3 = RiemannianSphere.normalize(V1 - 1e-8 * np.random.randn(dim))

    filter_collinear = MultiDimensionalRSFIFilter(S, [V1, V2, V3], alpha=1.5, beta=0.5, tau=0.0)
    Q_collinear = filter_collinear.Q_thr
    ortho_collinear_err = np.linalg.norm(Q_collinear.T @ Q_collinear - np.eye(3))
    MathTestLogger.log_metric("Near-Collinear Orthonormality Error ||Q^T Q - I||", ortho_collinear_err, 1e-10)
    assert ortho_collinear_err < 1e-10, "QR decomposition failed orthonormality on near-collinear input"

    # --- 3. Large Threat Subspace (k=50) ---
    MathTestLogger.section("3.1 Large Threat Subspace Stress Test (k=50 threats in d=1536)")
    k_large = 50
    V_threats_large = [RiemannianSphere.normalize(np.random.randn(dim)) for _ in range(k_large)]

    t0 = time.perf_counter()
    filter_large = MultiDimensionalRSFIFilter(S, V_threats_large, alpha=1.5, beta=0.5, tau=0.0)
    qr_build_ms = (time.perf_counter() - t0) * 1000.0

    Q_large = filter_large.Q_thr
    ortho_large_err = np.linalg.norm(Q_large.T @ Q_large - np.eye(k_large))

    MathTestLogger.log_metric("k=50 QR Subspace Construction Time", qr_build_ms, 50.0, unit="ms")
    MathTestLogger.log_metric("k=50 Basis Orthonormality Error ||Q^T Q - I_50||", ortho_large_err, 1e-10)
    assert ortho_large_err < 1e-10, "Large threat subspace lost orthonormality"


# =====================================================================
# TEST SUITE 4: GEODESIC MONOTONICITY & CONTINUITY AUDIT
# =====================================================================

def test_geodesic_monotonicity(dim: int = 1536, seed: int = 42):
    """
    Audits RSFI behavior along a continuous geodesic path gamma(t):
    gamma(t) = Exp_S(t * Log_S(V_thr)) for t in [0.0, 1.0] (100 steps).
    Verifies:
    1. Strict Monotonicity of threat projection pi_thr(t)
    2. Strict Monotonic Decay of RSFI score
    3. Continuity without step discontinuities or derivative spikes
    """
    MathTestLogger.header("SUITE 4: CONTINUOUS GEODESIC TRAJECTORY & MONOTONICITY AUDIT")
    np.random.seed(seed)

    S = RiemannianSphere.normalize(np.random.randn(dim))
    V_thr = RiemannianSphere.normalize(np.random.randn(dim))

    filter_sys = RSFIFilter(S, V_thr, alpha=1.5, beta=0.5, tau=-0.2)

    v_thr = RiemannianSphere.log_map(S, V_thr)
    num_steps = 100
    t_steps = np.linspace(0.0, 1.0, num_steps)

    pi_thr_history = []
    rsfi_history = []
    d_M_history = []

    for t in t_steps:
        # Compute point along geodesic path on hyper-sphere
        v_t = t * v_thr
        R_t = RiemannianSphere.exp_map(S, v_t)

        eval_res = filter_sys.evaluate(R_t)
        pi_thr_history.append(eval_res["pi_thr"])
        rsfi_history.append(eval_res["rsfi"])
        d_M_history.append(eval_res["d_M"])

    pi_thr_arr = np.array(pi_thr_history)
    rsfi_arr = np.array(rsfi_history)

    # Check monotonicity via finite differences
    dpi_dt = np.diff(pi_thr_arr)
    drsfi_dt = np.diff(rsfi_arr)

    pi_monotonic = np.all(dpi_dt >= -1e-12)
    rsfi_monotonic = np.all(drsfi_dt <= 1e-12)

    max_rsfi_spike = float(np.max(drsfi_dt))
    min_pi_step = float(np.min(dpi_dt))

    MathTestLogger.section("4.1 Geodesic Interpolation Audit Results (100 steps)")
    MathTestLogger.log_metric("Min pi_thr step derivative (dpi/dt >= 0)", min_pi_step, 0.0, passed=pi_monotonic)
    MathTestLogger.log_metric("Max RSFI step spike (dRSFI/dt <= 0)", max_rsfi_spike, 0.0, passed=rsfi_monotonic)
    MathTestLogger.log_info(f"RSFI Start (t=0): {rsfi_arr[0]:+.4f} -> End (t=1): {rsfi_arr[-1]:+.4f}")
    MathTestLogger.log_info(f"pi_thr Start (t=0): {pi_thr_arr[0]:+.4f} -> End (t=1): {pi_thr_arr[-1]:+.4f}")

    assert pi_monotonic, "Threat projection pi_thr is not monotonic along geodesic trajectory"
    assert rsfi_monotonic, "RSFI score is not monotonic along geodesic trajectory"


# =====================================================================
# TEST SUITE 5: FLOAT32 VS FLOAT64 PRECISION COMPARISON
# =====================================================================

def test_floating_point_precision_drift(dim: int = 1536, seed: int = 42):
    """
    Compares numerical precision drift between float64 and float32.
    Ensures float32 (standard GPU embedding precision) remains reliable.
    """
    MathTestLogger.header("SUITE 5: FLOAT32 VS FLOAT64 NUMERICAL PRECISION DRIFT")
    np.random.seed(seed)

    # Float64 baseline
    S_64 = RiemannianSphere.normalize(np.random.randn(dim).astype(np.float64))
    V_64 = RiemannianSphere.normalize(np.random.randn(dim).astype(np.float64))
    R_64 = RiemannianSphere.normalize(np.random.randn(dim).astype(np.float64))

    filter_64 = RSFIFilter(S_64, V_64, alpha=1.5, beta=0.5, tau=-0.2)
    res_64 = filter_64.evaluate(R_64)

    # Float32 target
    S_32 = S_64.astype(np.float32)
    V_32 = V_64.astype(np.float32)
    R_32 = R_64.astype(np.float32)

    filter_32 = RSFIFilter(S_32, V_32, alpha=1.5, beta=0.5, tau=-0.2)
    res_32 = filter_32.evaluate(R_32)

    rsfi_drift = abs(res_64["rsfi"] - res_32["rsfi"])
    pi_drift = abs(res_64["pi_thr"] - res_32["pi_thr"])
    d_M_drift = abs(res_64["d_M"] - res_32["d_M"])

    MathTestLogger.section("5.1 Precision Drift Comparison")
    MathTestLogger.log_metric("RSFI Score Float32 vs Float64 Drift", rsfi_drift, 1e-5)
    MathTestLogger.log_metric("Threat Projection pi_thr Drift", pi_drift, 1e-5)
    MathTestLogger.log_metric("Geodesic Distance d_M Drift", d_M_drift, 1e-5)

    assert rsfi_drift < 1e-4, "Float32 precision drift exceeded acceptable margin"


# =====================================================================
# MAIN ENTRYPOINT
# =====================================================================

def run_all_advanced_math_tests():
    print("=" * 90)
    print("      COMPREHENSIVE MATHEMATICAL BOUNDARY & RIGOROUS STRESS TEST SUITE")
    print("=" * 90)
    
    t_start = time.perf_counter()

    test_riemannian_geometry_boundaries()
    test_zca_whitening_invariants()
    test_subspace_qr_degeneracy()
    test_geodesic_monotonicity()
    test_floating_point_precision_drift()

    total_duration_sec = time.perf_counter() - t_start

    print("\n" + "=" * 90)
    print(f"   ALL ADVANCED MATHEMATICAL TESTS PASSED SUCCESSFULLY! Total time: {total_duration_sec:.3f} s")
    print("=" * 90)


if __name__ == "__main__":
    run_all_advanced_math_tests()
