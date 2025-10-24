import argparse
import time
import csv
from pathlib import Path
import importlib.util
import concurrent.futures
from collections import Counter

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

def _process_one(img_path_str, dlib_path_str):
    import cv2, importlib.util, time
    from pathlib import Path
    p = Path(img_path_str)
    img = cv2.imread(str(p))
    if img is None:
        return {"file": p.name, "status": "error", "error": "read_failed", "time": 0.0, "predicted": "", "confidence": "", "ground_truth": ""}

    # load dlib module once per worker
    if not hasattr(_process_one, "_mod_cache"):
        _process_one._mod_cache = {}
    key = dlib_path_str
    if key not in _process_one._mod_cache:
        spec = importlib.util.spec_from_file_location("dlib_eval_mod", dlib_path_str)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        _process_one._mod_cache[key] = mod
    dlib_mod = _process_one._mod_cache[key]

    # find callable
    func = getattr(dlib_mod, "recognize_face_in_crop_optimized", None)
    if func is None:
        # try heuristics
        for n in dir(dlib_mod):
            if "recognize" in n.lower() and callable(getattr(dlib_mod, n)):
                func = getattr(dlib_mod, n)
                break
    if func is None:
        return {"file": p.name, "status": "error", "error": "no_recognize_func", "time": 0.0, "predicted": "", "confidence": "", "ground_truth": ""}

    ok, res, dur, err = safe_call(func, img, img, (0,0,img.shape[1], img.shape[0]))
    # extract ground truth from filename (heuristic same as other evaluator)
    gt = ""
    base = p.stem
    if "_" in base:
        first = base.split("_", 1)[0]
        if any(c.isalpha() for c in first):
            gt = first
    if not ok:
        return {"file": p.name, "status": "error", "error": err, "time": dur, "predicted": "", "confidence": "", "ground_truth": gt}
    # normalize result if dict-like
    if isinstance(res, dict):
        pred = res.get("name") or res.get("recognized_name") or res.get("predicted","Unknown")
        conf = res.get("confidence", 0.0) or 0.0
    else:
        pred, conf = str(res), 0.0
    return {"file": p.name, "status": "ok", "error": "", "time": dur, "predicted": pred, "confidence": float(conf), "ground_truth": gt}

def main():
    p = argparse.ArgumentParser()
    p.add_argument(
        "--crops",
        default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\face_recognition\face-recognition-system\data\face_crops",
        help="Folder with face crop images (default points to face-recognition-system/data/face_crops)"
    )
    p.add_argument("--dlib", default=str(Path(__file__).resolve().parents[2] / "Dlibs CNN" / "dilib_cnn_main_optimized.py"),
                   help="Path to dlib module (default points to face_recognition/Dlibs CNN/...)")
    p.add_argument(
        "--out",
        default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\face_recognition\face-recognition-system\data\logs",
        help="Output folder for CSV logs and graphs"
    )
    p.add_argument("--limit", type=int, default=0, help="Number of crops to test (0 = all)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    crops_dir = Path(args.crops)
    if not crops_dir.exists():
        print("Crops folder not found:", crops_dir)
        return
    dlib_path = Path(args.dlib)
    if not dlib_path.exists():
        print("Dlib module not found:", dlib_path)
        return

    img_paths = sorted([p for p in crops_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if args.limit and args.limit > 0:
        img_paths = img_paths[: args.limit]
    total = len(img_paths)
    if total == 0:
        print("No images found in", crops_dir)
        return

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    out_csv = out_dir / f"dlib_results_{int(time.time())}.csv"

    completed = 0
    results = []
    print(f"Running Dlib-only evaluation on {total} images with {args.workers} workers...")

    # stats accumulator for summary
    model_name = "Dlib"
    stats = {"total": 0, "recognized": 0, "times": [], "confs": [], "names": Counter(), "correct": 0}

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_process_one, str(p), str(dlib_path)): p for p in img_paths}
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                row = {"file": str(futures[fut].name), "status":"error", "error": str(e), "time":0.0, "predicted":"", "confidence":""}
            completed += 1
            # print progress like: [5/50] filename time=0.268s OK
            status_text = "OK" if row.get("status") == "ok" else f"ERROR:{row.get('error')}"
            t = float(row.get("time", 0.0) or 0.0)
            print(f"[{completed}/{total}] {row.get('file','')} time={t:.3f}s {status_text}")
            results.append(row)

            # update stats
            stats["total"] += 1
            if row.get("status") == "ok":
                stats["times"].append(t)
                try:
                    conf = float(row.get("confidence", 0.0) or 0.0)
                except Exception:
                    conf = 0.0
                gt = row.get("ground_truth", "") or ""
                if row.get("predicted") and row.get("predicted") != "Unknown":
                    stats["recognized"] += 1
                    stats["confs"].append(conf)
                    stats["names"][row.get("predicted")] += 1
                # correct if ground truth present and predicted matches it
                try:
                    if gt and row.get("predicted") == gt:
                        stats["correct"] += 1
                except Exception:
                    pass

    # save CSV
    keys = ["file","status","predicted","confidence","time","ground_truth","error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k,"") for k in keys})

    # Print evaluation summary
    print("\n=== EVALUATION SUMMARY ===\n")
    avg_time = (sum(stats["times"]) / len(stats["times"])) if stats["times"] else 0.0
    avg_conf = (sum(stats["confs"]) / len(stats["confs"])) if stats["confs"] else 0.0
    top_names = stats["names"].most_common(10)

    print(f"Model: {model_name}")
    print(f"  Processed: {stats['total']}")
    recognized = stats["recognized"]
    pct = (recognized / stats["total"] * 100) if stats["total"] else 0.0
    print(f"  Recognized (not 'Unknown'): {recognized} ({pct:.1f}%)")
    # accuracy vs ground truth (requires ground_truth detected by filename)
    correct = stats.get("correct", 0)
    acc_pct = (correct / stats["total"] * 100.0) if stats["total"] else 0.0
    print(f"  Correct (predicted == ground_truth): {correct} ({acc_pct:.1f}%)")
    print(f"  Avg time / image: {avg_time:.4f}s")
    print(f"  Avg confidence (recognized): {avg_conf:.3f}")
    print(f"  Top predicted names: {top_names}")

    print(f"\nDetailed results saved to: {out_csv}")

if __name__ == "__main__":
    main()