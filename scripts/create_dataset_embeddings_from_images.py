import argparse
from pathlib import Path
import numpy as np
import torch
import cv2
from facenet_pytorch import InceptionResnetV1, MTCNN

def extract_face(mtcnn, img):
    try:
        boxes, _ = mtcnn.detect(img)
        if boxes is None or len(boxes) == 0:
            return None
        x1,y1,x2,y2 = [int(round(v)) for v in boxes[0]]
        h,w = img.shape[:2]
        x1, x2 = max(0, min(x1, w-1)), max(0, min(x2, w-1))
        y1, y2 = max(0, min(y1, h-1)), max(0, min(y2, h-1))
        if x2 <= x1 or y2 <= y1:
            return None
        return img[y1:y2, x1:x2]
    except Exception:
        return None

def preprocess_and_embed(embedder, img, device, size=160):
    try:
        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        face = cv2.resize(rgb, (size, size))
        arr = (face.astype('float32') - 127.5) / 128.0
        arr = np.transpose(arr, (2,0,1))
        t = torch.from_numpy(arr).unsqueeze(0).to(device)
        with torch.no_grad():
            out = embedder(t)
        emb = out.detach().cpu().numpy().reshape(-1)
        return emb / (np.linalg.norm(emb) + 1e-10)
    except Exception:
        return None

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--dataset-root", required=False,
                   default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\datasets\faces")
    p.add_argument("--out", required=False, default="", help="optional output folder (defaults to dataset-root)")
    p.add_argument("--min-images", type=int, default=1, help="min images per person to include")
    p.add_argument("--use-gpu", action="store_true")
    args = p.parse_args()

    dataset_root = Path(args.dataset_root)
    if not dataset_root.exists():
        print("Dataset root not found:", dataset_root); return

    out_root = Path(args.out) if args.out else dataset_root
    out_root.mkdir(parents=True, exist_ok=True)

    device = "cuda" if (args.use_gpu and torch.cuda.is_available()) else "cpu"
    print("Device:", device)
    mtcnn = MTCNN(keep_all=False, device=device, post_process=False)
    embedder = InceptionResnetV1(pretrained="vggface2").to(device).eval()

    names = []
    embs = []

    for person_dir in sorted(dataset_root.iterdir()):
        if not person_dir.is_dir():
            continue
        imgs = [p for p in person_dir.glob("*") if p.suffix.lower() in (".jpg",".jpeg",".png")]
        if len(imgs) < args.min_images:
            print(f"Skipping {person_dir.name} (only {len(imgs)} images)")
            continue
        person_embs = []
        for img_p in imgs:
            img = cv2.imread(str(img_p))
            if img is None:
                continue
            face = extract_face(mtcnn, img)
            if face is None:
                # fallback: try whole image if cropping failed
                face = img
            emb = preprocess_and_embed(embedder, face, device)
            if emb is not None:
                person_embs.append(emb)
        if len(person_embs) == 0:
            print(f"No valid embeddings for {person_dir.name}, skipping")
            continue
        centroid = np.mean(np.stack(person_embs, axis=0), axis=0)
        centroid = centroid / (np.linalg.norm(centroid) + 1e-10)
        names.append(person_dir.name)
        embs.append(centroid)
        print(f"Added {person_dir.name}: {len(person_embs)} images -> centroid")

    if len(names) == 0:
        print("No persons processed. Exiting.")
        return

    embs = np.vstack(embs).astype(np.float32)
    names_arr = np.array(names).astype(str)

    np.save(str(out_root / "embeddings.npy"), embs)
    np.save(str(out_root / "names.npy"), names_arr)
    print("Saved embeddings and names to:", out_root)

if __name__ == "__main__":
    main()