import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import base64

def encode_base64(text):
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def minkowski_dot(X, Y):
    # X, Y shape (N, d+1)
    # return -X0*Y0 + X1*Y1 + ...
    return -X[:, 0] * Y[:, 0] + np.sum(X[:, 1:] * Y[:, 1:], axis=1)

def hyperboloid_map(emb, c=1.0):
    # Map euclidean embedding to hyperboloid H^d
    # Scale embedding by c (curvature factor)
    scaled = emb * c
    x0 = np.sqrt(1 + np.sum(scaled**2, axis=1, keepdims=True))
    return np.hstack([x0, scaled])

def hyperbolic_distance(X, Y):
    dot = minkowski_dot(X, Y)
    # Clip to -1 to avoid nan in arccosh
    dot = np.clip(dot, a_min=None, a_max=-1.0)
    return np.arccosh(-dot)

def main():
    print("Loading model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    csv_path = Path(__file__).resolve().parent.parent / "data/results/sfi_wild_10k_results.csv"
    if not csv_path.exists():
        print("Data not found!")
        return
        
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels = (df["scenario_type"] == "MALICIOUS").astype(int).values
    
    obf_texts = [encode_base64(t) for t, l in zip(texts, labels) if l == 1][:200]
    safe_texts = [t for t, l in zip(texts, labels) if l == 0][:500]
    threat_texts = [t for t, l in zip(texts, labels) if l == 1][:500]
    
    print("Encoding...")
    emb_safe = model.encode(safe_texts, convert_to_numpy=True)
    emb_threat = model.encode(threat_texts, convert_to_numpy=True)
    emb_obf = model.encode(obf_texts, convert_to_numpy=True)
    
    sys_prompt = "You are a helpful AI assistant. You must always prioritize safety and refuse malicious requests."
    emb_S = model.encode([sys_prompt], convert_to_numpy=True)
    
    # Try different curvature scaling factors
    for c in [0.1, 0.5, 1.0, 2.0]:
        print(f"\n--- Curvature c={c} ---")
        H_safe = hyperboloid_map(emb_safe, c)
        H_threat = hyperboloid_map(emb_threat, c)
        H_obf = hyperboloid_map(emb_obf, c)
        H_S = hyperboloid_map(emb_S, c)
        
        # S is broadcasted
        H_S_repeat_safe = np.repeat(H_S, len(H_safe), axis=0)
        H_S_repeat_threat = np.repeat(H_S, len(H_threat), axis=0)
        H_S_repeat_obf = np.repeat(H_S, len(H_obf), axis=0)
        
        dist_safe = hyperbolic_distance(H_S_repeat_safe, H_safe)
        dist_threat = hyperbolic_distance(H_S_repeat_threat, H_threat)
        dist_obf = hyperbolic_distance(H_S_repeat_obf, H_obf)
        
        print(f"Safe Distance:   {np.mean(dist_safe):.4f} +- {np.std(dist_safe):.4f}")
        print(f"Threat Distance: {np.mean(dist_threat):.4f} +- {np.std(dist_threat):.4f}")
        print(f"Obf Distance:    {np.mean(dist_obf):.4f} +- {np.std(dist_obf):.4f}")
        
        # Calculate AUC for Safe vs Threat based purely on distance
        # labels: Safe=0, Threat=1. Distance should be higher for threat.
        labels_arr = np.concatenate([np.zeros(len(dist_safe)), np.ones(len(dist_threat))])
        scores_arr = np.concatenate([dist_safe, dist_threat])
        from sklearn.metrics import roc_auc_score
        auc = roc_auc_score(labels_arr, scores_arr)
        print(f"AUC (Safe vs Threat via Hyperbolic Distance): {auc:.4f}")

if __name__ == "__main__":
    main()
