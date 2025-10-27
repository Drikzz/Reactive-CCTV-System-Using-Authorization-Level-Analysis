import os
import sys
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
import importlib.util
import concurrent.futures

# --- helpers to load modules by path ---
def load_module_from_path(name, path):
    spec = importlib.util.spec_from_file_location(name, str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def safe_call(func, *args, **kwargs):
    t0 = time.time()
    try:
        res = func(*args, **kwargs)
        dur = time.time() - t0
        return True, res, dur, None
    except Exception as e:
        return False, None, time.time() - t0, str(e)

def parse_ground_truth_from_filename(fname):
    # If files are named like "NAME_xxx.jpg" return NAME as gt
    base = Path(fname).stem
    if "_" in base:
        first = base.split("_", 1)[0]
        # simple heuristic: treat as GT if contains letters and not just video stem
        if any(c.isalpha() for c in first):
            return first
    return None

def ensure_models_trained():
    """Ensure models are trained with dataset faces before evaluation"""
    dataset_path = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\datasets\faces"
    
    print("Checking if models are trained with dataset faces...")
    
    # Check if FaceNet is trained
    facenet_path = Path(__file__).resolve().parents[3] / "face_recognition" / "Facenet" / "facenet_main.py"
    try:
        facenet_mod = load_module_from_path("facenet_main", facenet_path)
        if hasattr(facenet_mod, 'classifier') and facenet_mod.classifier is not None:
            print("✅ FaceNet appears to be trained")
        else:
            print("❌ FaceNet not trained - run training first")
            return False
    except Exception as e:
        print(f"❌ FaceNet loading failed: {e}")
        return False
    
    return True

def main():
    # Add this check at the beginning of main()
    if not ensure_models_trained():
        print("ERROR: Models not trained. Please train models with your dataset first.")
        print("Run the training scripts in face_recognition/Facenet/, face_recognition/ArcFace/, etc.")
        return
    
    parser = argparse.ArgumentParser(description="Run FaceNet / ArcFace / Dlib on saved face crops and log predictions")
    parser.add_argument("--crops", default=str(Path(__file__).resolve().parents[1] / "data" / "face_crops"),
                        help="Folder with face crop images")
    parser.add_argument("--out", default=str(Path(__file__).resolve().parents[1] / "data" / "logs"),
                        help="Output folder for CSV logs")
    parser.add_argument("--facenet", help="Path to facenet_main.py (if omitted, auto-locate)", default=None)
    parser.add_argument("--arcface", help="Path to arcface_main.py (if omitted, auto-locate)", default=None)
    # parser.add_argument("--dlib", help="Path to dlib main (if omitted, auto-locate)", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for evaluation")
    parser.add_argument("--use-gpu", action="store_true", help="Attempt to run recognizers on GPU (passes device hints to modules / child processes)")
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Load optional ground-truth mapping (filename -> name), prefer CSV when available
    gt_map = {}
    gt_csv = crops_dir / "ground_truth.csv"
    if gt_csv.exists():
        try:
            with open(gt_csv, newline="", encoding="utf-8") as gf:
                rdr = csv.reader(gf)
                for r in rdr:
                    if len(r) >= 2:
                        fn = r[0].strip()
                        name = r[1].strip()
                        if fn:
                            gt_map[fn] = name
                            gt_map[Path(fn).stem] = name
        except Exception:
            pass

    def _norm(s):
        """Normalize labels for robust comparison: keep only alphanumeric, lowercase."""
        if s is None:
            return ""
        s = str(s)
        return "".join(c.lower() for c in s if c.isalnum())

    if not crops_dir.exists():
        print("ERROR: crops folder not found:", crops_dir)
        return

    # autolocate modules if not provided (based on your repo layout)
    repo_root = Path(__file__).resolve().parents[3]  # project root
    if args.facenet:
        facenet_path = Path(args.facenet)
    else:
        facenet_path = repo_root / "face_recognition" / "Facenet" / "facenet_main.py"
    if args.arcface:
        arcface_path = Path(args.arcface)
    else:
        arcface_path = repo_root / "face_recognition" / "ArcFace" / "arcface_main.py"
    # Dlib intentionally skipped here; use scripts/evaluate_dlib_on_crops.py to run Dlib separately
    dlib_path = None

    print("Using modules:")
    print(" FaceNet:", facenet_path)
    print(" ArcFace:", arcface_path)
    print(" Dlib  : skipped (use evaluate_dlib_on_crops.py)")

    # If user requested GPU, export an env flag so child processes see it
    if args.use_gpu:
        os.environ["FORCE_GPU"] = "1"
        try:
            import torch
            if not torch.cuda.is_available():
                print("[WARN] --use-gpu requested but torch reports no CUDA available. Child processes may still run on CPU.")
            else:
                print("[INFO] CUDA available. Child workers will attempt to use GPU.")
        except Exception:
            print("[WARN] torch not available in main process; child processes will detect CUDA if installed.")

    # load modules (may print their own logs)
    facenet_mod = None
    arcface_mod = None
    dlib_mod = None

    try:
        facenet_mod = load_module_from_path("facenet_main", facenet_path)
        # hint to module to prefer CUDA if possible
        if args.use_gpu and facenet_mod is not None:
            try:
                import torch
                if hasattr(facenet_mod, "set_device"):
                    facenet_mod.set_device("cuda")
                setattr(facenet_mod, "DEVICE", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            except Exception:
                pass
    except Exception as e:
        print("WARNING: could not load FaceNet module:", e)

    try:
        arcface_mod = load_module_from_path("arcface_main", arcface_path)
        # hint to module to prefer CUDA if possible
        if args.use_gpu and arcface_mod is not None:
            try:
                import torch
                if hasattr(arcface_mod, "set_device"):
                    arcface_mod.set_device("cuda")
                setattr(arcface_mod, "DEVICE", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            except Exception:
                pass
    except Exception as e:
        print("WARNING: could not load ArcFace module:", e)

    # Dlib loading removed from this combined evaluator
    dlib_mod = None

    # Prepare ArcFaceSystem instance if available
    arcface_system = None
    if arcface_mod and hasattr(arcface_mod, "ArcFaceSystem"):
        try:
            # try to pass device hint to ArcFaceSystem if supported
            if args.use_gpu:
                try:
                    arcface_system = arcface_mod.ArcFaceSystem(use_yolo=False, device="cuda")
                except TypeError:
                    # fallback to constructor without device arg
                    arcface_system = arcface_mod.ArcFaceSystem(use_yolo=False)
            else:
                arcface_system = arcface_mod.ArcFaceSystem(use_yolo=False)
        except Exception as e:
            print("WARNING: ArcFaceSystem init failed:", e)
            arcface_system = None

    # Prepare predictor callables
    predictors = []

    # FaceNet: module-level function recognize_face_in_crop(person_crop, original_frame, person_bbox)
    if facenet_mod and hasattr(facenet_mod, "recognize_face_in_crop"):
        def facenet_wrapper(img):
            # facenet expects person_crop and original_frame + person_bbox
            h, w = img.shape[:2]
            ok, res, dt, err = safe_call(facenet_mod.recognize_face_in_crop, img, img, (0,0,w,h))
            if not ok:
                return {"status":"error","error":err}
            return {"status":"ok","result":res,"time":dt}
        predictors.append(("FaceNet", facenet_wrapper))
    else:
        print("FaceNet function not available; skipping FaceNet")

    # ArcFace: use arcface_system.recognize_face_in_crop_enhanced(person_crop, original_frame, person_bbox, frame_num)
    if arcface_system and hasattr(arcface_system, "recognize_face_in_crop_enhanced"):
        def arcface_wrapper(img):
            h,w = img.shape[:2]
            ok, res, dt, err = safe_call(arcface_system.recognize_face_in_crop_enhanced, img, img, (0,0,w,h), 0)
            if not ok:
                return {"status":"error","error":err}
            return {"status":"ok","result":res,"time":dt}
        predictors.append(("ArcFace", arcface_wrapper))
    else:
        print("ArcFace recognizer not available; skipping ArcFace")

    # Dlib: module-level function recognize_face_in_crop_optimized(person_crop, original_frame, person_bbox)
    # Dlib evaluation removed from this script
    print("Dlib evaluation skipped for this run.")

    if not predictors:
        print("ERROR: no recognizers available. Exiting.")
        return

    # iterate images
    img_paths = sorted([p for p in crops_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if args.limit:
        img_paths = img_paths[:args.limit]

    results = []
    # track 'correct' (predicted == ground_truth) per model
    model_stats = {name: {"total":0, "recognized":0, "times":[],"confs":[],"names":{}, "correct": 0} for name, _ in predictors}

    # use process pool to evaluate images in parallel
    workers = min(getattr(args, "workers", 4), max(1, os.cpu_count() or 1))
    print(f"Processing {len(img_paths)} images with {workers} worker processes...")

    # progress counters per model (used to print e.g. "[50/50] filename time=0.268s OK" for Dlib)
    total_images = len(img_paths)
    model_progress = {name: 0 for name, _ in predictors}

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
        # map each image to worker (pass model file paths so workers can import)
        facenet_path_str = str(facenet_path) if facenet_mod else ""
        arcface_path_str = str(arcface_path) if arcface_mod else ""
        dlib_path_str = ""
        futures = {exe.submit(_process_image_worker, str(p), facenet_path_str, arcface_path_str, dlib_path_str): p for p in img_paths}
        for fut in concurrent.futures.as_completed(futures):
            try:
                rows = fut.result()
            except Exception as e:
                print("Worker failed:", e)
                continue
            for row in rows:
                # try to get ground truth if available
                # prefer CSV mapping, then stem mapping only (avoid video-stem heuristic)
                gt = gt_map.get(row["file"]) or gt_map.get(Path(row["file"]).stem)
                # if no explicit GT, fall back to dataset-based assignment when available
                ds = (row.get("dataset_predicted") or row.get("nearest_dataset") or row.get("dataset_match") or "") 
                if not gt and ds:
                    gt = ds
                row["ground_truth"] = gt or ""

                # update aggregate stats
                mn = row["model"]
                model_stats[mn]["total"] += 1
                if row["status"] == "ok" and row["predicted"] and row["predicted"] != "Unknown":
                    model_stats[mn]["recognized"] += 1
                    model_stats[mn]["confs"].append(float(row.get("confidence", 0.0) or 0.0))
                    model_stats[mn]["names"].setdefault(row["predicted"], 0)
                    model_stats[mn]["names"][row["predicted"]] += 1
                # count correct predictions when ground truth is present
                gt = row.get("ground_truth", "") or ""
                try:
                    if gt and row["status"] == "ok" and row.get("predicted") and _norm(row.get("predicted")) == _norm(gt):
                        model_stats[mn]["correct"] += 1
                except Exception:
                    pass
                try:
                    model_stats[mn]["times"].append(float(row.get("time", 0.0) or 0.0))
                except Exception:
                    pass
                results.append(row)

                # Print progress for Dlib (and keep counters for other models if needed)
                # (Dlib removed) progress is tracked in aggregate stats only

    # save detailed CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"model_comparison_results_{timestamp}.csv"
    keys = ["file","model","status","predicted","confidence","time","ground_truth","error"]
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k,"") for k in keys})

    # print summary per model
    print("\n=== EVALUATION SUMMARY ===")
    for model_name, stats in model_stats.items():
        total = stats["total"]
        recognized = stats["recognized"] 
        avg_time = (sum(stats["times"]) / len(stats["times"])) if stats["times"] else 0.0
        avg_conf = (sum(stats["confs"]) / len(stats["confs"])) if stats["confs"] else 0.0
        most_common = sorted(stats["names"].items(), key=lambda x: x[1], reverse=True)[:5]
        correct = stats.get("correct", 0)
        acc_pct = (correct / total * 100.0) if total else 0.0
        print(f"\nModel: {model_name}")
        print(f"  Processed: {total}")
        print(f"  Recognized (not 'Unknown'): {recognized} ({(recognized/total*100) if total else 0:.1f}%)")
        print(f"  Correct (predicted == ground_truth): {correct} ({acc_pct:.1f}%)")
        # accuracy over images that have ground-truth labels
        rows_with_gt = [r for r in results if _norm(r.get("ground_truth")) and r.get("model")==model_name]
        num_with_gt = len(rows_with_gt)
        correct_with_gt = sum(1 for r in rows_with_gt if _norm(r.get("predicted")) == _norm(r.get("ground_truth")))
        if num_with_gt:
            print(f"  Accuracy (over {num_with_gt} with GT): {correct_with_gt}/{num_with_gt} ({(correct_with_gt/num_with_gt*100):.1f}%)")
        print(f"  Avg time / image: {avg_time:.4f}s")
        print(f"  Avg confidence (recognized): {avg_conf:.3f}")
        print(f"  Top predicted names: {most_common}")
    print(f"\nDetailed results saved to: {out_csv}")

def _process_image_worker(img_path_str, facenet_path_str, arcface_path_str, dlib_path_str):
    """
    Worker executed in subprocess: loads modules once per process, runs recognizers for one image,
    returns list of result rows (same dict shape used in main).
    """
    from pathlib import Path
    import importlib.util
    import cv2
    import time
    import os

    img_p = Path(img_path_str)
    img = cv2.imread(str(img_p))
    if img is None:
        return []

    # per-process module cache
    if not hasattr(_process_image_worker, "_mods"):
        _process_image_worker._mods = {}

    def _load_once(key, path_str):
        if not path_str:
            return None
        key2 = f"{key}:{path_str}"
        if key2 in _process_image_worker._mods:
            return _process_image_worker._mods[key2]
        spec = importlib.util.spec_from_file_location(key, path_str)
        mod = importlib.util.module_from_spec(spec)
        spec.loader.exec_module(mod)
        # if parent requested GPU, attempt to hint module to use CUDA
        try:
            if os.getenv("FORCE_GPU") == "1":
                try:
                    import torch
                    if hasattr(mod, "set_device"):
                        try:
                            mod.set_device("cuda")
                        except Exception:
                            pass
                    # set common attribute names used by modules
                    setattr(mod, "DEVICE", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                    setattr(mod, "device", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
                except Exception:
                    pass
        except Exception:
            pass
        _process_image_worker._mods[key2] = mod
        return mod

    facenet_mod = _load_once("facenet_main", facenet_path_str) if facenet_path_str else None
    arcface_mod = _load_once("arcface_main", arcface_path_str) if arcface_path_str else None
    dlib_mod = None

    rows = []
    # local safe_call (avoid cross-process dependency)
    def _safe_call_local(func, *a, **kw):
        t0 = time.time()
        try:
            res = func(*a, **kw)
            return True, res, time.time() - t0, None
        except Exception as e:
            return False, None, time.time() - t0, str(e)

    # run FaceNet
    if facenet_mod and hasattr(facenet_mod, "recognize_face_in_crop"):
        ok, res, dt, err = _safe_call_local(facenet_mod.recognize_face_in_crop, img, img, (0,0,img.shape[1], img.shape[0]))
        if not ok:
            rows.append({
                "file": img_p.name, "model": "FaceNet", "status":"error", "error": err,
                "predicted":"", "confidence":"", "time": dt, "ground_truth": ""
            })
        else:
            # normalize result dict -> predicted/confidence
            if isinstance(res, dict):
                pred = res.get("name") or res.get("recognized_name") or res.get("predicted","Unknown")
                conf = res.get("confidence") or 0.0
            else:
                pred, conf = (str(res), 0.0)
            rows.append({
                "file": img_p.name, "model": "FaceNet", "status":"ok", "error":[],
                "predicted": pred, "confidence": float(conf), "time": float(dt), "ground_truth": ""
            })

    # run ArcFace
    if arcface_mod and hasattr(arcface_mod, "ArcFaceSystem"):
        try:
            # create system instance per-process if needed and cache it
            if not hasattr(_process_image_worker, "_arc_sys"):
                try:
                    # honor module-level DEVICE hint if available
                    dev = getattr(arcface_mod, "DEVICE", None)
                    if dev is not None:
                        try:
                            _process_image_worker._arc_sys = arcface_mod.ArcFaceSystem(use_yolo=False, device=str(dev))
                        except TypeError:
                            _process_image_worker._arc_sys = arcface_mod.ArcFaceSystem(use_yolo=False)
                    else:
                        _process_image_worker._arc_sys = arcface_mod.ArcFaceSystem(use_yolo=False)
                except Exception:
                    _process_image_worker._arc_sys = None
            arc_sys = _process_image_worker._arc_sys
            if arc_sys and hasattr(arc_sys, "recognize_face_in_crop_enhanced"):
                ok, res, dt, err = _safe_call_local(arc_sys.recognize_face_in_crop_enhanced, img, img, (0,0,img.shape[1], img.shape[0]), 0)
                if not ok:
                    rows.append({"file": img_p.name, "model": "ArcFace", "status":"error", "error": err, "predicted":"", "confidence":"", "time": dt, "ground_truth": ""})
                else:
                    if isinstance(res, dict):
                        pred = res.get("name") or res.get("recognized_name") or res.get("predicted","Unknown")
                        conf = res.get("confidence") or 0.0
                    else:
                        pred, conf = (str(res), 0.0)
                    rows.append({"file": img_p.name, "model": "ArcFace", "status":"ok", "error":"", "predicted": pred, "confidence": float(conf), "time": float(dt), "ground_truth": ""})
        except Exception as e:
            rows.append({"file": img_p.name, "model": "ArcFace", "status":"error", "error": str(e), "predicted":"", "confidence":"", "time":0.0, "ground_truth": ""})

    return rows

if __name__ == "__main__":
    main()