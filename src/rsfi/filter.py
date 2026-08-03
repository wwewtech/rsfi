"""
Riemannian System Fidelity Index (RSFI) filters.
"""

from typing import Any, Dict, List, Union
import numpy as np

from rsfi.geometry import RiemannianSphere


class RSFIFilter:
    """
    Single-Vector Tangent Space System Fidelity Filter.
    Evaluates loyalty/safety of responses with respect to system prompt S and threat vector V_thr.
    """

    def __init__(
        self,
        S: np.ndarray,
        V_thr: np.ndarray,
        alpha: float = 1.5,
        beta: float = 0.5,
        tau: float = -0.2,
    ):
        self.S = S
        self.V_thr = V_thr
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        # Map threat anchor into tangent space T_S M and compute basis unit vector e_thr
        self.v_thr = RiemannianSphere.log_map(S, V_thr)
        self.norm_v_thr = np.linalg.norm(self.v_thr)
        self.e_thr = self.v_thr / np.maximum(self.norm_v_thr, 1e-15)

    def evaluate(self, R_t: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate response vector R_t in tangent space T_S M.

        Returns dictionary containing:
        - v_R: Log-map vector in T_S M
        - d_M: Geodesic distance d_M(S, R_t)
        - pi_thr: Projection onto threat direction e_thr
        - v_perp: Orthogonal safe component vector
        - norm_v_perp: Magnitude of safe component
        - rsfi: Calculated RSFI score
        - action: "PASS" or "BLOCK" decision
        """
        # 1. Log map of response R_t in T_S M
        v_R = RiemannianSphere.log_map(self.S, R_t)
        d_M = float(np.linalg.norm(v_R))

        # 2. Projection onto threat direction e_thr
        pi_thr = float(np.dot(v_R, self.e_thr))

        # 3. Orthogonal component
        v_perp = v_R - pi_thr * self.e_thr
        norm_v_perp = float(np.linalg.norm(v_perp))

        # 4. RSFI calculation
        rsfi = norm_v_perp - self.alpha * pi_thr - self.beta * d_M
        action = "PASS" if rsfi >= self.tau else "BLOCK"

        return {
            "v_R": v_R,
            "d_M": d_M,
            "pi_thr": pi_thr,
            "v_perp": v_perp,
            "norm_v_perp": norm_v_perp,
            "rsfi": rsfi,
            "action": action,
        }


class MultiDimensionalRSFIFilter:
    """
    Generalized Subspace Tangent System Fidelity Filter.
    Constructs an orthonormal basis Q_thr of a k-dimensional threat subspace
    via QR decomposition to detect multi-vector zero-day attacks.
    """

    def __init__(
        self,
        S: np.ndarray,
        V_threats: Union[List[np.ndarray], np.ndarray],
        alpha: float = 1.0,
        beta: float = 0.5,
        tau: float = 0.0,
        is_tangent: bool = False,
    ):
        self.S = S
        self.alpha = alpha
        self.beta = beta
        self.tau = tau

        # 1. Map threat vectors into tangent space T_S M if they aren't already tangent vectors
        U_tangent = []
        for V in V_threats:
            if is_tangent:
                v_thr = V
            else:
                v_thr = RiemannianSphere.log_map(S, V)
            U_tangent.append(v_thr)

        U_matrix = np.column_stack(U_tangent)  # Shape: (dim, k)

        # 2. Build orthonormal basis via QR decomposition
        Q_thr, _ = np.linalg.qr(U_matrix)
        self.Q_thr = Q_thr

    def evaluate(self, R_t: np.ndarray) -> Dict[str, Any]:
        """
        Evaluate response vector R_t against k-dimensional threat subspace.
        """
        # 1. Log map of response to T_S M
        v_R = RiemannianSphere.log_map(self.S, R_t)
        d_M = float(np.linalg.norm(v_R))

        # 2. Projection onto threat subspace (Proj_U = Q * Q^T * v)
        coeffs = self.Q_thr.T @ v_R
        v_proj = self.Q_thr @ coeffs
        norm_proj = float(np.linalg.norm(v_proj))

        # 3. Orthogonal component
        v_perp = v_R - v_proj
        norm_v_perp = float(np.linalg.norm(v_perp))

        # 4. Multi-dimensional RSFI calculation
        rsfi = norm_v_perp - self.alpha * norm_proj - self.beta * d_M
        action = "PASS" if rsfi >= self.tau else "BLOCK"

        return {
            "v_R": v_R,
            "d_M": d_M,
            "norm_proj": norm_proj,
            "norm_v_perp": norm_v_perp,
            "rsfi": rsfi,
            "action": action,
        }
