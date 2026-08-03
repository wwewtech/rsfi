"""
Unit tests for RiemannianSphere geometry module.
"""

import numpy as np

from rsfi.geometry import RiemannianSphere


def test_normalization():
    v = np.array([3.0, 4.0, 0.0])
    v_norm = RiemannianSphere.normalize(v)
    assert np.isclose(np.linalg.norm(v_norm), 1.0)


def test_geodesic_distance():
    x = np.array([1.0, 0.0, 0.0])
    y = np.array([0.0, 1.0, 0.0])
    dist = RiemannianSphere.geodesic_distance(x, y)
    assert np.isclose(dist, np.pi / 2)


def test_log_map_axioms():
    dim = 128
    np.random.seed(42)
    S = RiemannianSphere.normalize(np.random.randn(dim))

    for _ in range(50):
        R = RiemannianSphere.normalize(np.random.randn(dim))
        d_geom = RiemannianSphere.geodesic_distance(S, R)
        v_R = RiemannianSphere.log_map(S, R)
        norm_v_R = np.linalg.norm(v_R)

        # 1. Norm of Log_S(R) must equal geodesic distance d_M(S, R)
        assert np.isclose(norm_v_R, d_geom, atol=1e-10)

        # 2. Tangent vector v_R must be orthogonal to anchor S (<S, v_R> == 0)
        assert np.isclose(np.dot(S, v_R), 0.0, atol=1e-10)


def test_exp_map():
    dim = 64
    np.random.seed(123)
    S = RiemannianSphere.normalize(np.random.randn(dim))
    R = RiemannianSphere.normalize(np.random.randn(dim))

    v_R = RiemannianSphere.log_map(S, R)
    R_recovered = RiemannianSphere.exp_map(S, v_R)

    assert np.isclose(np.linalg.norm(R_recovered), 1.0)
    assert np.allclose(R, R_recovered, atol=1e-7)
