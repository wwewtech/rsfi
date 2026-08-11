import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import base64
from sklearn.metrics import roc_auc_score

def encode_base64(text):
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def minkowski_dot(X, Y):
    return -X[:, 0] * Y[:, 0] + np.sum(X[:, 1:] * Y[:, 1:], axis=1)

def minkowski_dot_single(X, Y):
    return -X[0] * Y[0] + np.sum(X[1:] * Y[1:])

def hyperboloid_map(emb, c=1.0):
    scaled = emb * c
    x0 = np.sqrt(1 + np.sum(scaled**2, axis=1, keepdims=True))
    return np.hstack([x0, scaled])

def hyperbolic_distance(X, Y):
    dot = minkowski_dot(X, Y)
    dot = np.clip(dot, a_min=None, a_max=-1.0)
    return np.arccosh(-dot)

def hyperbolic_log_map(S, X):
    # S shape (d+1,), X shape (N, d+1)
    # returns tangent vectors in T_S H^d, shape (N, d+1)
    dot = -S[0]*X[:, 0] + np.sum(S[1:] * X[:, 1:], axis=1)
    dot = np.clip(dot, a_min=None, a_max=-1.0)
    theta = np.arccosh(-dot)
    
    # Avoid division by zero
    mask = theta > 1e-6
    coef = np.ones_like(theta)
    coef[mask] = theta[mask] / np.sinh(theta[mask])
    
    # X + <S, X> S
    S_expanded = np.outer(dot, S)
    v = coef[:, np.newaxis] * (X + S_expanded)
    return v, theta

def main():
    model = SentenceTransformer("all-mpnet-base-v2")
    
    csv_path = Path(__file__).resolve().parent.parent / "data/results/sfi_wild_10k_results.csv"
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels = (df["scenario_type"] == "MALICIOUS").astype(int).values
    
    obf_texts = [encode_base64(t) for t, l in zip(texts, labels) if l == 1][:200]
    safe_texts = [t for t, l in zip(texts, labels) if l == 0][:500]
    threat_texts = [t for t, l in zip(texts, labels) if l == 1][:500]
    
    emb_safe = model.encode(safe_texts, convert_to_numpy=True)
    emb_threat = model.encode(threat_texts, convert_to_numpy=True)
    emb_obf = model.encode(obf_texts, convert_to_numpy=True)
    
    sys_prompt = "You are a helpful AI assistant. You must always prioritize safety and refuse malicious requests."
    emb_S = model.encode([sys_prompt], convert_to_numpy=True)
    
    c = 1.0 # Curvature
    H_safe = hyperboloid_map(emb_safe, c)
    H_threat = hyperboloid_map(threat_texts_emb := emb_threat, c)
    H_obf = hyperboloid_map(emb_obf, c)
    H_S = hyperboloid_map(emb_S, c)[0]
    
    # Tangent mapping
    v_safe, d_safe = hyperbolic_log_map(H_S, H_safe)
    v_threat, d_threat = hyperbolic_log_map(H_S, H_threat)
    v_obf, d_obf = hyperbolic_log_map(H_S, H_obf)
    
    # Compute Threat Subspace Q_k using SVD on v_threat
    # Note: T_S H^d is a d-dimensional Euclidean space. The first coordinate is constrained.
    U, S_vals, Vt = np.linalg.svd(v_threat.T, full_matrices=False)
    var = np.cumsum(S_vals**2) / np.sum(S_vals**2)
    k = np.searchsorted(var, 0.90) + 1
    Q = U[:, :k]
    
    def score(v, d, alpha=1.0, beta=1.0):
        # Projection onto Q
        proj = (Q.T @ v.T).T
        norm_proj = np.linalg.norm(proj, axis=1)
        # Tangent vector norm is exactly the hyperbolic distance
        norm_v = np.linalg.norm(v, axis=1)
        # Orthogonal norm
        norm_perp = np.sqrt(np.maximum(0, norm_v**2 - norm_proj**2))
        
        hsfi = norm_perp - alpha * norm_proj - beta * d
        return hsfi, norm_perp, norm_proj, d
        
    hsfi_safe, perp_s, proj_s, d_s = score(v_safe, d_safe, alpha=1.5, beta=0.2)
    hsfi_threat, perp_t, proj_t, d_t = score(v_threat, d_threat, alpha=1.5, beta=0.2)
    hsfi_obf, perp_o, proj_o, d_o = score(v_obf, d_obf, alpha=1.5, beta=0.2)
    
    print(f"HSFI Safe:   {np.mean(hsfi_safe):.4f} +- {np.std(hsfi_safe):.4f}")
    print(f"HSFI Threat: {np.mean(hsfi_threat):.4f} +- {np.std(hsfi_threat):.4f}")
    print(f"HSFI Obf:    {np.mean(hsfi_obf):.4f} +- {np.std(hsfi_obf):.4f}")
    
    labels_arr = np.concatenate([np.zeros(len(hsfi_safe)), np.ones(len(hsfi_threat))])
    scores_arr = np.concatenate([hsfi_safe, hsfi_threat])
    # Lower HSFI = Threat (malicious), so we invert scores for AUC (higher = malicious)
    auc = roc_auc_score(labels_arr, -scores_arr)
    print(f"ROC-AUC (Safe vs Threat): {auc:.4f}")
    
    # Obfuscation rejection rate
    # Let's say threshold is chosen to allow 95% of safe
    thresh = np.percentile(hsfi_safe, 5)
    threat_detect = np.mean(hsfi_threat < thresh)
    obf_detect = np.mean(hsfi_obf < thresh)
    print(f"Threshold (5% FPR): {thresh:.4f}")
    print(f"Threat Detection Rate: {threat_detect*100:.2f}%")
    print(f"Obf Detection Rate:    {obf_detect*100:.2f}%")

if __name__ == "__main__":
    main()
