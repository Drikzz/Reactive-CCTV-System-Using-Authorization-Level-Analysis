import csv
from pathlib import Path
import re
import numpy as np

ROOT = Path(r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis")
CROPS_DIR = ROOT / "face_recognition" / "face-recognition-system" / "data" / "face_crops"
DATASET_DIR = ROOT / "datasets" / "faces"
OUT_CSV = CROPS_DIR / "ground_truth.csv"

def load_dataset_names(dataset_dir):
    names_npy = dataset_dir / "names.npy"
    if names_npy.exists():
        return [str(x) for x in np.load(str(names_npy)).astype(str)]
    if dataset_dir.exists():
        return [p.name for p in sorted(dataset_dir.iterdir()) if p.is_dir()]
    return []

def normalize_token(s):
    return "".join(ch.lower() for ch in s if ch.isalnum())

def filename_tokens(fn):
    base = Path(fn).stem
    return [t for t in re.split(r'[^0-9A-Za-z]+', base) if t]

def main():
    dataset_names = load_dataset_names(DATASET_DIR)
    norm_to_name = {normalize_token(n): n for n in dataset_names}
    if not CROPS_DIR.exists():
        print("Crops dir not found:", CROPS_DIR); return
    files = sorted([p.name for p in CROPS_DIR.glob("*") if p.suffix.lower() in (".jpg",".png")])
    rows = []
    found = 0
    for fn in files:
        gt = ""
        for tok in filename_tokens(fn):
            nrm = normalize_token(tok)
            if nrm in norm_to_name:
                gt = norm_to_name[nrm]
                found += 1
                break
        rows.append((fn, gt))
    OUT_CSV.parent.mkdir(parents=True, exist_ok=True)
    with OUT_CSV.open("w", newline="", encoding="utf-8") as fh:
        w = csv.writer(fh)
        w.writerow(["file","name"])
        for fn, gt in rows:
            w.writerow([fn, gt])
    print(f"Wrote {OUT_CSV} ({len(rows)} rows, {found} mapped to dataset labels)")

if __name__ == "__main__":
    main()