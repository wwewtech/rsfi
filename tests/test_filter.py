"""
Unit tests for RSFIFilter and MultiDimensionalRSFIFilter.
"""

import numpy as np

from rsfi.geometry import RiemannianSphere
from rsfi.filter import RSFIFilter, MultiDimensionalRSFIFilter


def test_rsfi_pythagoras_decomposition():
    dim = 256
    np.random.seed(42)

    S = RiemannianSphere.normalize(np.random.randn(dim))
    V_thr = RiemannianSphere.normalize(np.random.randn(dim))

    filter_sys = RSFIFilter(S, V_thr, alpha=1.5, beta=0.5, tau=-0.2)

    for _ in range(20):
        R = RiemannianSphere.normalize(np.random.randn(dim))
        res = filter_sys.evaluate(R)

        # Verify Pythagorean identity in tangent space T_S M:
        # ||v_R||^2 = ||v_perp||^2 + pi_thr^2
        v_R_sq = res["d_M"] ** 2
        decomp_sq = res["norm_v_perp"] ** 2 + res["pi_thr"] ** 2

        assert np.isclose(v_R_sq, decomp_sq, atol=1e-10)

        # Verify orthogonality between v_perp and threat basis e_thr
        assert np.isclose(np.dot(res["v_perp"], filter_sys.e_thr), 0.0, atol=1e-10)


def test_multidimensional_rsfi_filter():
    dim = 128
    np.random.seed(2026)

    S = RiemannianSphere.normalize(np.random.randn(dim))
    threat_1 = RiemannianSphere.normalize(np.random.randn(dim))
    threat_2 = RiemannianSphere.normalize(np.random.randn(dim))
    threat_3 = RiemannianSphere.normalize(np.random.randn(dim))

    V_threats = [threat_1, threat_2, threat_3]
    multi_filter = MultiDimensionalRSFIFilter(
        S, V_threats, alpha=1.5, beta=0.5, tau=0.0
    )

    # Check orthonormal basis Q shape and orthogonality (Q^T * Q == I_k)
    k = len(V_threats)
    assert multi_filter.Q_thr.shape == (dim, k)
    assert np.allclose(multi_filter.Q_thr.T @ multi_filter.Q_thr, np.eye(k), atol=1e-10)

    R = RiemannianSphere.normalize(np.random.randn(dim))
    res = multi_filter.evaluate(R)

    assert "rsfi" in res
    assert res["action"] in ["PASS", "BLOCK"]
