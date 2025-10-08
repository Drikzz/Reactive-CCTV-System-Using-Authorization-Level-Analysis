import os, cv2, numpy as np, joblib, time
import torch
from pprint import pprint
# ensure repo root is on sys.path so local package imports work when running this file directly
import sys
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))
from face_recognition.ArcFace.arcface_main import SimpleArcFaceRecognizer, MODELS_DIR, IMAGE_SIZE

def run():
    r = SimpleArcFaceRecognizer()
    ok = r.load_models()
    print("models_loaded:", ok)
    if not ok:
        return

    print("\n[INFO] Torch device:", torch.device("cuda" if torch.cuda.is_available() else "cpu"))
    print("[INFO] Embedder device:", next(r.embedder.parameters()).device)
    print("\n[INFO] Classifier.classes_:")
    pprint(getattr(r.classifier, "classes_", None))
    print("\n[INFO] LabelEncoder.classes_:")
    try:
        pprint(getattr(r.encoder, "classes_", None))
    except Exception as e:
        print("  (encoder load error)", e)
    print("\n[INFO] Centroids present:", r.centroids is not None)
    if r.centroids is not None:
        print(" Centroid keys sample:", list(r.centroids.keys())[:10])

    ds = "datasets/faces"
    people = [d for d in os.listdir(ds) if os.path.isdir(os.path.join(ds, d))]
    if not people:
        print("No people folders in datasets/faces")
        return

    # sample up to 3 people, up to 3 images each
    samples = []
    for p in people[:3]:
        files = [f for f in os.listdir(os.path.join(ds, p)) if f.lower().endswith(('.jpg','.png','.jpeg'))]
        for f in files[:3]:
            samples.append((p, os.path.join(ds, p, f)))

    print(f"\n[INFO] Running recognition on {len(samples)} sample images...\n")
    for expected, path in samples:
        img = cv2.imread(path)
        if img is None:
            print("  Failed to read", path); continue

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        tensor = r.transform(img_rgb).unsqueeze(0).to(next(r.embedder.parameters()).device)
        with torch.no_grad():
            emb = r.embedder(tensor).cpu().numpy().flatten()
        norm_before = np.linalg.norm(emb)
        if norm_before > 0:
            emb_n = emb / (norm_before + 1e-10)
        else:
            emb_n = emb

        # classifier output
        if hasattr(r.classifier, "predict_proba"):
            probs = r.classifier.predict_proba([emb_n])[0]
            top_idx = int(np.argmax(probs))
            top_prob = float(probs[top_idx])
        else:
            pred = r.classifier.predict([emb_n])[0]
            probs = None
            top_prob = 1.0
            top_idx = list(r.classifier.classes_).index(pred)

        encoded_label = r.classifier.classes_[top_idx]
        try:
            decoded = r.encoder.inverse_transform([encoded_label])[0]
        except Exception:
            try:
                decoded = r.encoder.classes_[int(encoded_label)]
            except Exception:
                decoded = str(encoded_label)

        centroid_dist = None
        if r.centroids is not None:
            c = r.centroids.get(decoded)
            if c is not None:
                centroid_dist = float(np.linalg.norm(emb_n - np.asarray(c, dtype=np.float32)))

        print(f"Image: {path}")
        print(f"  expected: {expected}")
        print(f"  emb_norm(before)={norm_before:.4f} mean={emb_n.mean():.4f} std={emb_n.std():.4f}")
        print(f"  encoded_label={encoded_label} decoded={decoded} top_prob={top_prob:.4f} centroid_dist={centroid_dist}")
        if probs is not None:
            top5 = sorted(list(enumerate(probs)), key=lambda x: x[1], reverse=True)[:5]
            print(f"  top5 (idx,prob): {top5}")
        print("")

    # quick intra/inter distance check for first two people (if available)
    if len(people) >= 2:
        p0 = people[0]
        p1 = people[1]
        def get_embs(person, n=5):
            imgs = [f for f in os.listdir(os.path.join(ds, person)) if f.lower().endswith(('.jpg','.png'))][:n]
            out = []
            for f in imgs:
                img = cv2.imread(os.path.join(ds, person, f))
                if img is None: continue
                t = r.transform(cv2.cvtColor(img, cv2.COLOR_BGR2RGB)).unsqueeze(0).to(next(r.embedder.parameters()).device)
                with torch.no_grad():
                    e = r.embedder(t).cpu().numpy().flatten()
                e = e / (np.linalg.norm(e) + 1e-10)
                out.append(e)
            return np.stack(out) if len(out)>0 else None

        e0 = get_embs(p0, n=5)
        e1 = get_embs(p1, n=5)
        if e0 is not None and e1 is not None:
            intra0 = np.mean([np.linalg.norm(a-b) for i,a in enumerate(e0) for j,b in enumerate(e0) if i<j])
            intra1 = np.mean([np.linalg.norm(a-b) for i,a in enumerate(e1) for j,b in enumerate(e1) if i<j])
            inter01 = np.mean([np.linalg.norm(a-b) for a in e0 for b in e1])
            print(f"\nDistance summary ({p0} vs {p1}): intra_{p0}={intra0:.4f}, intra_{p1}={intra1:.4f}, inter={inter01:.4f}")

if __name__ == "__main__":
    run()