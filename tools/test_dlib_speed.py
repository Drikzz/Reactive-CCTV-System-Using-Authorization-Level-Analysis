import time
import argparse
from pathlib import Path
import importlib.util
import cv2

def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def safe_call(func, *args, **kwargs):
    t0 = time.time()
    try:
        res = func(*args, **kwargs)
        return True, res, time.time() - t0, None
    except Exception as e:
        return False, None, time.time() - t0, str(e)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", default=str(Path(__file__).resolve().parents[1] / "face_recognition" / "face-recognition-system" / "data" / "face_crops"),
                        help="Folder with face crop images")
    parser.add_argument("--dlib", default=str(Path(__file__).resolve().parents[1] / "face_recognition" / "Dlibs CNN" / "dilib_cnn_main_optimized.py"),
                        help="Path to dlib module file")
    parser.add_argument("--limit", type=int, default=50, help="Number of crops to test")
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    if not crops_dir.exists():
        print("Crops folder not found:", crops_dir)
        return

    dlib_path = Path(args.dlib)
    if not dlib_path.exists():
        print("Dlib module not found at:", dlib_path)
        return

    print("Loading dlib module from:", dlib_path)
    dlib_mod = load_module_from_path("dlib_test_mod", dlib_path)
    if not hasattr(dlib_mod, "recognize_face_in_crop_optimized"):
        # try common exported name fallback
        candidates = [n for n in dir(dlib_mod) if "recognize" in n.lower()]
        print("recognize_face_in_crop_optimized not found. Available candidates:", candidates)
        return

    func = getattr(dlib_mod, "recognize_face_in_crop_optimized")
    img_paths = sorted([p for p in crops_dir.glob("*") if p.suffix.lower() in [".jpg", ".jpeg", ".png"]])[:args.limit]
    if not img_paths:
        print("No images found in:", crops_dir)
        return

    times = []
    errors = 0
    print(f"Testing {len(img_paths)} crops sequentially using {func.__name__} ...")
    for i, p in enumerate(img_paths, 1):
        img = cv2.imread(str(p))
        if img is None:
            print(f"[{i}] failed to read {p.name}, skipping")
            continue
        ok, res, dur, err = safe_call(func, img, img, (0,0,img.shape[1], img.shape[0]))
        times.append(dur)
        status = "OK" if ok else f"ERROR: {err}"
        print(f"[{i}/{len(img_paths)}] {p.name} time={dur:.3f}s {status}")
        if not ok:
            errors += 1

    total = sum(times)
    avg = total / len(times) if times else 0.0
    print("\n=== Dlib speed test summary ===")
    print(f"Images processed: {len(times)}")
    print(f"Errors: {errors}")
    print(f"Total time: {total:.3f}s")
    print(f"Average time / image: {avg:.3f}s")
    print(f"Median time / image: {sorted(times)[len(times)//2]:.3f}s")

if __name__ == "__main__":
    main()