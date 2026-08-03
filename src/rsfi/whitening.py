"""
Anisotropy correction and ZCA/PCA Spherical Whitening module.
"""

import numpy as np
from scipy.linalg import sqrtm

from rsfi.geometry import RiemannianSphere


class SphericalWhitening:
    """ZCA Whitening with re-projection onto the hyper-sphere S^(d-1)."""

    def __init__(self, dim: int, reg: float = 1e-6):
        self.dim = dim
        self.reg = reg
        self.mu = None
        self.W = None

    def fit(self, data: np.ndarray):
        """
        Compute empirical mean and ZCA whitening matrix W = Sigma^(-1/2).

        Args:
            data: Background calibration data array of shape (N, dim).
        """
        self.mu = np.mean(data, axis=0)
        centered = data - self.mu
        cov = np.cov(centered, rowvar=False)

        # Regularization for numerical stability
        cov += np.eye(self.dim) * self.reg

        # Inverse square root of covariance matrix
        inv_sqrt_cov = sqrtm(np.linalg.inv(cov))
        self.W = np.real(inv_sqrt_cov)

    def transform(self, x: np.ndarray) -> np.ndarray:
        """
        Apply whitening transformation and re-normalize onto S^(d-1).

        Args:
            x: Input vector or matrix of shape (dim,) or (N, dim).
        """
        if self.mu is None or self.W is None:
            raise RuntimeError("SphericalWhitening must be fitted before transform.")

        centered = x - self.mu
        whitened = np.dot(centered, self.W)
        return RiemannianSphere.normalize(whitened)
