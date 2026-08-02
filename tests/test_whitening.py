"""
Unit tests for SphericalWhitening module.
"""

import numpy as np

from rsfi.whitening import SphericalWhitening


def test_spherical_whitening_fit_transform():
    dim = 64
    num_samples = 200
    np.random.seed(42)

    # Generate anisotropic synthetic background data
    scale = np.random.uniform(0.5, 5.0, size=dim)
    raw_data = np.random.randn(num_samples, dim) * scale

    whitening = SphericalWhitening(dim=dim)
    whitening.fit(raw_data)

    test_vec = raw_data[0]
    transformed = whitening.transform(test_vec)

    # Output must be L2 normalized on sphere S^(d-1)
    assert np.isclose(np.linalg.norm(transformed), 1.0)


def test_unfitted_transform_raises():
    whitening = SphericalWhitening(dim=32)
    raised = False
    try:
        whitening.transform(np.zeros(32))
    except RuntimeError:
        raised = True
    assert raised, "Unfitted transform did not raise RuntimeError"
