"""
Riemannian Geometry utilities for hyper-spheres S^(d-1).
"""

import numpy as np


class RiemannianSphere:
    """Computations on the unit Riemannian hyper-sphere S^(d-1)."""

    @staticmethod
    def normalize(v: np.ndarray) -> np.ndarray:
        """L2-normalize vector or batch of vectors along the last axis."""
        norm = np.linalg.norm(v, axis=-1, keepdims=True)
        return v / np.maximum(norm, 1e-15)

    @staticmethod
    def geodesic_distance(x: np.ndarray, y: np.ndarray) -> float:
        """Geodesic distance d_M(x, y) = arccos(<x, y>)."""
        dot = np.clip(np.dot(x, y), -1.0, 1.0)
        return float(np.arccos(dot))

    @classmethod
    def log_map(cls, x: np.ndarray, y: np.ndarray) -> np.ndarray:
        """
        Riemannian logarithmic map Log_x(y): S^(d-1) -> T_x S^(d-1).
        Projects point y on the sphere into the tangent space at point x.
        """
        theta = cls.geodesic_distance(x, y)
        if theta < 1e-12:
            return np.zeros_like(x)

        # Orthogonal projection onto tangent space T_x S^(d-1)
        proj = y - np.dot(x, y) * x
        norm_proj = np.linalg.norm(proj)

        if norm_proj < 1e-12:
            return np.zeros_like(x)

        # Scale vector norm to geodesic distance theta
        return (theta / norm_proj) * proj

    @classmethod
    def exp_map(cls, x: np.ndarray, v: np.ndarray) -> np.ndarray:
        """
        Riemannian exponential map Exp_x(v): T_x S^(d-1) -> S^(d-1).
        Maps tangent vector v back onto the hyper-sphere.
        """
        norm_v = np.linalg.norm(v)
        if norm_v < 1e-12:
            return x / np.linalg.norm(x)

        direction = v / norm_v
        y = np.cos(norm_v) * x + np.sin(norm_v) * direction
        return cls.normalize(y)
