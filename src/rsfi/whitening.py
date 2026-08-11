"""
Anisotropy correction and ZCA/PCA Spherical Whitening module.

ИСПРАВЛЕНО (аудит):
Старая версия регуляризовала ковариацию константой reg=1e-6 независимо от N/d.
В заявленной "сильной стороне" метода — few-shot режиме N_ref<=5 на d=768+ —
у эмпирической ковариации реально ~N-1 ненулевых степеней свободы в d-мерном пространстве;
константа 1e-6 не спасает обращение почти вырожденной матрицы.
Заменено на Ledoit-Wolf shrinkage (Ledoit & Wolf, 2004) — коэффициент регуляризации
подбирается аналитически по N и d, а не фиксируется руками.
"""

import numpy as np
from scipy.linalg import sqrtm
from sklearn.covariance import LedoitWolf

from rsfi.geometry import RiemannianSphere


class SphericalWhitening:
    """ZCA Whitening with re-projection onto the hyper-sphere S^(d-1)."""

    def __init__(self, dim: int, reg: float = 1e-6, method: str = "ledoit_wolf"):
        """
        Args:
            dim: размерность эмбеддинга.
            reg: используется только если method="fixed" (старое поведение).
            method: "ledoit_wolf" (по умолчанию, рекомендуется для N_ref <= dim)
                     или "fixed" (старое поведение).
        """
        self.dim = dim
        self.reg = reg
        self.method = method
        self.mu = None
        self.W = None
        self.shrinkage_: float | None = None

    def fit(self, data: np.ndarray):
        """
        Compute empirical mean and ZCA whitening matrix W = Sigma^(-1/2).

        Args:
            data: Background calibration data array of shape (N, dim).
        """
        n_samples = data.shape[0]
        self.mu = np.mean(data, axis=0)
        centered = data - self.mu

        if self.method == "ledoit_wolf":
            lw = LedoitWolf().fit(centered)
            cov = lw.covariance_
            self.shrinkage_ = float(lw.shrinkage_)
            if n_samples <= self.dim:
                print(
                    f"  [SphericalWhitening] WARNING: N_ref={n_samples} <= dim={self.dim}. "
                    f"Ledoit-Wolf shrinkage={self.shrinkage_:.4f} (чем ближе к 1, тем "
                    f"сильнее регуляризация доминирует над эмпирической ковариацией)."
                )
        elif self.method == "fixed":
            cov = np.cov(centered, rowvar=False)
            cov += np.eye(self.dim) * self.reg
            self.shrinkage_ = None
        else:
            raise ValueError(f"Unknown method: {self.method}")

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
