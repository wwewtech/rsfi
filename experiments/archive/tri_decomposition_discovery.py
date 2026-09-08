import sys
from pathlib import Path
import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer
import base64

sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
from rsfi.geometry import RiemannianSphere
from rsfi.whitening import SphericalWhitening

def encode_base64(text):
    return base64.b64encode(text.encode('utf-8')).decode('utf-8')

def main():
    print("Loading model...")
    model = SentenceTransformer("all-mpnet-base-v2")
    
    # Load some safe and threat data
    csv_path = Path(__file__).resolve().parent.parent / "data/results/sfi_wild_10k_results.csv"
    if not csv_path.exists():
        print("Data not found!")
        return
        
    df = pd.read_csv(csv_path)
    texts = df["text"].tolist()
    labels = (df["scenario_type"] == "MALICIOUS").astype(int).values
    
    # Generate obfuscated attacks
    print("Generating obfuscated threats...")
    obf_texts = [encode_base64(t) for t, l in zip(texts, labels) if l == 1][:200]
    
    # Let's take samples
    safe_texts = [t for t, l in zip(texts, labels) if l == 0][:500]
    threat_texts = [t for t, l in zip(texts, labels) if l == 1][:500]
    
    print("Encoding...")
    emb_safe = RiemannianSphere.normalize(model.encode(safe_texts, convert_to_numpy=True))
    emb_threat = RiemannianSphere.normalize(model.encode(threat_texts, convert_to_numpy=True))
    emb_obf = RiemannianSphere.normalize(model.encode(obf_texts, convert_to_numpy=True))
    
    sys_prompt = "You are a helpful AI assistant."
    S = RiemannianSphere.normalize(model.encode([sys_prompt], convert_to_numpy=True))[0]
    
    # ZCA Whitening on all natural text
    all_natural = np.vstack([emb_safe, emb_threat])
    zca = SphericalWhitening(dim=768)
    zca.fit(all_natural)
    
    emb_safe_w = zca.transform(emb_safe)
    emb_threat_w = zca.transform(emb_threat)
    emb_obf_w = zca.transform(emb_obf)
    S_w = zca.transform(S.reshape(1, -1))[0]
    
    # Tangent mapping
    print("Mapping to tangent space...")
    def to_tangent(embs):
        return np.array([RiemannianSphere.log_map(S_w, e) for e in embs])
        
    v_safe = to_tangent(emb_safe_w)
    v_threat = to_tangent(emb_threat_w)
    v_obf = to_tangent(emb_obf_w)
    
    # 1. Build Language Subspace Q_lang from all natural language
    v_all_lang = np.vstack([v_safe, v_threat])
    U_l, S_l, Vt_l = np.linalg.svd(v_all_lang.T, full_matrices=False)
    # Pick k_L to explain 95% variance
    var_l = np.cumsum(S_l**2) / np.sum(S_l**2)
    k_L = np.searchsorted(var_l, 0.95) + 1
    Q_lang = U_l[:, :k_L]
    print(f"Language Subspace Dimension (k_L): {k_L} / 768")
    
    # 2. Build Threat Subspace Q_thr INSIDE Q_lang
    # First, project threats into Q_lang
    v_threat_lang = (Q_lang.T @ v_threat.T).T  # shape (N_threat, k_L)
    U_t, S_t, Vt_t = np.linalg.svd(v_threat_lang.T, full_matrices=False)
    var_t = np.cumsum(S_t**2) / np.sum(S_t**2)
    k_T = np.searchsorted(var_t, 0.90) + 1
    Q_thr_local = U_t[:, :k_T]  # inside Q_lang
    Q_thr = Q_lang @ Q_thr_local  # back to 768 space
    print(f"Threat Subspace Dimension (k_T): {k_T} / {k_L}")
    
    # 3. Analyze decomposition
    def analyze(v_set, name):
        # void component: orthogonal to Q_lang
        v_lang = (Q_lang @ (Q_lang.T @ v_set.T)).T
        v_void = v_set - v_lang
        norm_void = np.linalg.norm(v_void, axis=1)
        
        # threat component: projection onto Q_thr
        v_thr = (Q_thr @ (Q_thr.T @ v_set.T)).T
        norm_thr = np.linalg.norm(v_thr, axis=1)
        
        # safe component: inside Q_lang, orthogonal to Q_thr
        v_safe_comp = v_lang - v_thr
        norm_safe = np.linalg.norm(v_safe_comp, axis=1)
        
        print(f"\n{name} Analysis:")
        print(f"  ||v_void||: {np.mean(norm_void):.4f} +- {np.std(norm_void):.4f}")
        print(f"  ||v_thr|| : {np.mean(norm_thr):.4f} +- {np.std(norm_thr):.4f}")
        print(f"  ||v_safe||: {np.mean(norm_safe):.4f} +- {np.std(norm_safe):.4f}")

    analyze(v_safe, "SAFE Texts")
    analyze(v_threat, "THREAT Texts")
    analyze(v_obf, "OBFUSCATED (Base64) Threats")

if __name__ == "__main__":
    main()
