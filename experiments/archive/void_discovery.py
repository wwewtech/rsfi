import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import base64
from sklearn.metrics import roc_auc_score

def encode_base64(text):
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def main():
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
    
    emb_safe = model.encode(safe_texts, convert_to_numpy=True)
    emb_threat = model.encode(threat_texts, convert_to_numpy=True)
    emb_obf = model.encode(obf_texts, convert_to_numpy=True)
    
    # Normalize
    emb_safe = emb_safe / np.linalg.norm(emb_safe, axis=1, keepdims=True)
    emb_threat = emb_threat / np.linalg.norm(emb_threat, axis=1, keepdims=True)
    emb_obf = emb_obf / np.linalg.norm(emb_obf, axis=1, keepdims=True)
    
    # 1. Build the Language Subspace Q_L (Strictly low rank)
    # We use Safe texts as the representative of standard language
    U_l, S_l, Vt_l = np.linalg.svd(emb_safe.T, full_matrices=False)
    k_L = 30 # Fixed strict low rank
    Q_L = U_l[:, :k_L]
    
    def get_energies(embs):
        # Projection onto Q_L (Language energy)
        proj_L = (Q_L.T @ embs.T).T
        energy_L = np.linalg.norm(proj_L, axis=1)**2
        
        # Void energy (Orthogonal to language)
        energy_void = 1.0 - energy_L # Since embs are unit vectors
        return energy_L, energy_void
        
    eL_safe, eV_safe = get_energies(emb_safe)
    eL_threat, eV_threat = get_energies(emb_threat)
    eL_obf, eV_obf = get_energies(emb_obf)
    
    print(f"Void Energy (Safe):   {np.mean(eV_safe):.4f} +- {np.std(eV_safe):.4f}")
    print(f"Void Energy (Threat): {np.mean(eV_threat):.4f} +- {np.std(eV_threat):.4f}")
    print(f"Void Energy (Obf):    {np.mean(eV_obf):.4f} +- {np.std(eV_obf):.4f}")

if __name__ == "__main__":
    main()
