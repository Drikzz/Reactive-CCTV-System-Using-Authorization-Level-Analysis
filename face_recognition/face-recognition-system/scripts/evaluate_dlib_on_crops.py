import argparse
import time
import csv
from pathlib import Path
import importlib.util
import concurrent.futures
from collections import Counter
from datetime import datetime

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

def _norm(s):
    """Normalize labels for robust comparison: keep only alphanumeric, lowercase."""
    if s is None:
        return ""
    s = str(s)
    return "".join(c.lower() for c in s if c.isalnum())

def build_ground_truth_from_dataset(dataset_path):
    """
    Build ground truth mapping from LevelsAuthorization dataset structure.
    Returns tuple: (gt_map, class_names)
    """
    gt_map = {}
    class_names = set()
    dataset_path = Path(dataset_path)
    
    if not dataset_path.exists():
        print(f"[WARN] Dataset path not found: {dataset_path}")
        return gt_map, class_names
    
    print(f"[INFO] Building ground truth from: {dataset_path}")
    
    person_count = 0
    image_count = 0
    
    # Top-level folders (treat each folder name as a class/label)
    for entry in dataset_path.iterdir():
        if entry.is_dir():
            class_names.add(entry.name)
    
    # Iterate through dataset to build file-level mapping
    for person_dir in dataset_path.iterdir():
        if not person_dir.is_dir():
            continue
        
        person_name = person_dir.name
        person_count += 1
        
        # Check for multi-angle structure (front/left/right subdirs)
        angle_dirs = [person_dir / angle for angle in ['front', 'left', 'right']]
        has_angles = any(d.exists() and d.is_dir() for d in angle_dirs)
        
        if has_angles:
            # Multi-angle structure
            for angle_dir in angle_dirs:
                if angle_dir.exists() and angle_dir.is_dir():
                    for img_file in angle_dir.glob("*"):
                        if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                            gt_map[img_file.name] = person_name
                            gt_map[img_file.stem] = person_name
                            image_count += 1
        else:
            # Flat structure
            for img_file in person_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    gt_map[img_file.name] = person_name
                    gt_map[img_file.stem] = person_name
                    image_count += 1
    
    print(f"[INFO] Ground truth built: {person_count} persons/classes, {image_count} images")
    print(f"[INFO] Class names found: {sorted(class_names)}")
    return gt_map, class_names

def match_crop_to_ground_truth(crop_filename, gt_map, class_names=None):
    """
    Match a face crop filename to ground truth.
    """
    import re
    crop_path = Path(crop_filename)
    stem = crop_path.stem
    name = crop_path.name
    
    # Strategy 1: Direct match
    if name in gt_map:
        return gt_map[name]
    
    # Strategy 2: Stem match
    if stem in gt_map:
        return gt_map[stem]
    
    # Strategy 3: Parse crop filename patterns
    if '_' in stem:
        first_part = stem.split('_')[0]
        # Check against class_names
        if class_names:
            for cls in class_names:
                if first_part.lower() == cls.lower():
                    return cls
        # Check against gt_map values
        for v in set(gt_map.values()):
            if first_part.lower() == v.lower():
                return v
    
    # Strategy 4: Substring match against class_names
    if class_names:
        stem_lower = stem.lower()
        for cls in sorted(class_names, key=lambda s: -len(s)):
            cls_clean = "".join(c.lower() for c in cls if c.isalnum())
            if cls.lower() in stem_lower or cls_clean in stem_lower:
                return cls
    
    # Strategy 5: Fuzzy match
    if class_names:
        stem_alnum = "".join(c.lower() for c in stem if c.isalnum())
        for cls in class_names:
            cls_alnum = "".join(c.lower() for c in cls if c.isalnum())
            if cls_alnum and cls_alnum in stem_alnum:
                return cls
    
    return None

def _process_one(img_path_str, dlib_path_str, gt_map_dict, class_names_list):
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
        for n in dir(dlib_mod):
            if "recognize" in n.lower() and callable(getattr(dlib_mod, n)):
                func = getattr(dlib_mod, n)
                break
    if func is None:
        return {"file": p.name, "status": "error", "error": "no_recognize_func", "time": 0.0, "predicted": "", "confidence": "", "ground_truth": ""}

    ok, res, dur, err = safe_call(func, img, img, (0,0,img.shape[1], img.shape[0]))
    
    # Extract ground truth using enhanced matching
    gt = match_crop_to_ground_truth(p.name, gt_map_dict, set(class_names_list) if class_names_list else None)
    if not gt:
        gt = ""
    
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
        help="Folder with face crop images"
    )
    p.add_argument(
        "--dataset",
        default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\datasets\LevelsAuthorization",
        help="Path to LevelsAuthorization dataset for ground truth"
    )
    p.add_argument("--dlib", default=str(Path(__file__).resolve().parents[2] / "Dlibs CNN" / "dilib_cnn_main_optimized.py"),
                   help="Path to dlib module")
    p.add_argument(
        "--out",
        default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\face_recognition\face-recognition-system\data\logs",
        help="Output folder for CSV logs"
    )
    p.add_argument("--limit", type=int, default=0, help="Number of crops to test (0 = all)")
    p.add_argument("--workers", type=int, default=4)
    args = p.parse_args()

    crops_dir = Path(args.crops)
    dataset_dir = Path(args.dataset)
    if not crops_dir.exists():
        print("Crops folder not found:", crops_dir)
        return
    dlib_path = Path(args.dlib)
    if not dlib_path.exists():
        print("Dlib module not found:", dlib_path)
        return

    # Build ground truth
    print("\n" + "="*80)
    print("BUILDING GROUND TRUTH FROM DATASET")
    print("="*80)
    gt_map, class_names = build_ground_truth_from_dataset(dataset_dir)

    img_paths = sorted([p for p in crops_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if args.limit and args.limit > 0:
        img_paths = img_paths[: args.limit]
    total = len(img_paths)
    if total == 0:
        print("No images found in", crops_dir)
        return

    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"dlib_results_{timestamp}.csv"

    completed = 0
    results = []
    print(f"\n{'='*80}")
    print(f"PROCESSING {total} IMAGES")
    print("="*80)
    print(f"Running Dlib evaluation with {args.workers} workers...\n")

    # stats accumulator for summary
    model_name = "Dlib"
    stats = {
        "total": 0, 
        "recognized": 0, 
        "times": [], 
        "confs": [], 
        "names": Counter(), 
        "correct": 0,
        "has_gt": 0,
        "correct_with_gt": 0,
        "class_stats": {}  # NEW: Per-class statistics
    }

    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_process_one, str(p), str(dlib_path), dict(gt_map), list(class_names)): p for p in img_paths}
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                row = {"file": str(futures[fut].name), "status":"error", "error": str(e), "time":0.0, "predicted":"", "confidence":"", "ground_truth":""}
            completed += 1
            
            if completed % 10 == 0:
                print(f"[PROGRESS] Processed {completed}/{total} images...")
            
            results.append(row)

            # update stats
            stats["total"] += 1
            gt = row.get("ground_truth", "") or ""
            pred = row.get("predicted", "")
            
            if row.get("status") == "ok":
                try:
                    t = float(row.get("time", 0.0) or 0.0)
                    stats["times"].append(t)
                except Exception:
                    pass
                
                try:
                    conf = float(row.get("confidence", 0.0) or 0.0)
                except Exception:
                    conf = 0.0
                
                if pred and pred != "Unknown":
                    stats["recognized"] += 1
                    stats["confs"].append(conf)
                    stats["names"][pred] += 1
                
                # Track accuracy
                if gt and gt != "Unknown":
                    stats["has_gt"] += 1
                    if _norm(pred) == _norm(gt):
                        stats["correct"] += 1
                        stats["correct_with_gt"] += 1
            
            # NEW: Track per-class statistics
            pred_norm = _norm(pred) if pred else ""
            gt_norm = _norm(gt) if gt else ""
            
            if gt and gt != "Unknown":
                # Initialize class stats if needed
                if gt not in stats["class_stats"]:
                    stats["class_stats"][gt] = {
                        "tp": 0, "fp": 0, "fn": 0,
                        "total_gt": 0, "total_pred": 0
                    }
                
                stats["class_stats"][gt]["total_gt"] += 1
                
                # True Positive: predicted correctly
                if pred_norm == gt_norm:
                    stats["class_stats"][gt]["tp"] += 1
                else:
                    # False Negative
                    stats["class_stats"][gt]["fn"] += 1
            
            # Track false positives
            if row.get("status") == "ok" and pred and pred != "Unknown":
                if pred not in stats["class_stats"]:
                    stats["class_stats"][pred] = {
                        "tp": 0, "fp": 0, "fn": 0,
                        "total_gt": 0, "total_pred": 0
                    }
                
                stats["class_stats"][pred]["total_pred"] += 1
                
                # If prediction doesn't match ground truth, it's a false positive
                if gt and gt != "Unknown" and pred_norm != gt_norm:
                    stats["class_stats"][pred]["fp"] += 1

    # save CSV
    keys = ["file","status","predicted","confidence","time","ground_truth","error","match"]
    
    # Add match column
    for r in results:
        if r.get("ground_truth") and r.get("ground_truth") != "Unknown":
            r["match"] = "✓" if _norm(r.get("predicted")) == _norm(r.get("ground_truth")) else "✗"
        else:
            r["match"] = "N/A"
    
    with open(out_csv, "w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k,"") for k in keys})

    print(f"\n{'='*80}")
    print("SAVING RESULTS")
    print("="*80)
    print(f"✅ Results saved to: {out_csv}")

    # Print evaluation summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    avg_time = (sum(stats["times"]) / len(stats["times"])) if stats["times"] else 0.0
    avg_conf = (sum(stats["confs"]) / len(stats["confs"])) if stats["confs"] else 0.0
    top_names = stats["names"].most_common(10)

    total = stats["total"]
    recognized = stats["recognized"]
    has_gt = stats["has_gt"]
    correct_with_gt = stats["correct_with_gt"]

    print(f"\n{'='*40}")
    print(f"Model: {model_name}")
    print(f"{'='*40}")
    print(f"  Total images processed: {total}")
    print(f"  Images with ground truth: {has_gt}")
    print(f"  Recognized (not 'Unknown'): {recognized} ({(recognized/total*100) if total else 0:.1f}%)")
    
    if has_gt > 0:
        accuracy = (correct_with_gt / has_gt) * 100
        print(f"  ✅ ACCURACY: {correct_with_gt}/{has_gt} ({accuracy:.1f}%)")
        
        if recognized > 0:
            precision = (correct_with_gt / recognized) * 100
            print(f"  Precision: {precision:.1f}%")
            
            recall = (correct_with_gt / has_gt) * 100
            print(f"  Recall: {recall:.1f}%")
            
            if precision > 0 and recall > 0:
                f1 = 2 * (precision * recall) / (precision + recall)
                print(f"  F1 Score: {f1:.1f}%")
    else:
        print(f"  ⚠️  No ground truth available for accuracy calculation")
    
    print(f"  Average time per image: {avg_time:.4f}s")
    print(f"  Average confidence: {avg_conf:.3f}")
    print(f"  Top predictions: {top_names}")

    # NEW: Per-class metrics
    if stats["class_stats"]:
        print(f"\n  {'─'*38}")
        print(f"  PER-CLASS METRICS:")
        print(f"  {'─'*38}")
        
        sorted_classes = sorted(stats["class_stats"].keys())
        
        for class_name in sorted_classes:
            class_data = stats["class_stats"][class_name]
            tp = class_data["tp"]
            fp = class_data["fp"]
            fn = class_data["fn"]
            total_gt = class_data["total_gt"]
            total_pred = class_data["total_pred"]
            
            print(f"\n  Class: {class_name}")
            print(f"  {'-'*36}")
            print(f"    Ground truth samples: {total_gt}")
            print(f"    Predicted as this class: {total_pred}")
            print(f"    True Positives: {tp}")
            print(f"    False Positives: {fp}")
            print(f"    False Negatives: {fn}")
            
            if total_gt > 0:
                class_accuracy = (tp / total_gt) * 100
                print(f"    ✅ Accuracy: {tp}/{total_gt} ({class_accuracy:.1f}%)")
            else:
                print(f"    ✅ Accuracy: N/A (no ground truth samples)")
            
            if total_pred > 0:
                class_precision = (tp / total_pred) * 100
                print(f"    Precision: {class_precision:.1f}%")
            else:
                print(f"    Precision: N/A (never predicted)")
            
            if total_gt > 0:
                class_recall = (tp / total_gt) * 100
                print(f"    Recall: {class_recall:.1f}%")
            else:
                print(f"    Recall: N/A (no ground truth samples)")
            
            if total_pred > 0 and total_gt > 0:
                if class_precision > 0 and class_recall > 0:
                    class_f1 = 2 * (class_precision * class_recall) / (class_precision + class_recall)
                    print(f"    F1 Score: {class_f1:.1f}%")
                else:
                    print(f"    F1 Score: 0.0%")

    # NEW: Save per-class metrics to separate CSV
    per_class_csv = out_dir / f"dlib_per_class_metrics_{timestamp}.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "class", "total_gt", "total_pred", "tp", "fp", "fn", 
                     "accuracy", "precision", "recall", "f1_score"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        
        for class_name, class_data in sorted(stats["class_stats"].items()):
            tp = class_data["tp"]
            fp = class_data["fp"]
            fn = class_data["fn"]
            total_gt = class_data["total_gt"]
            total_pred = class_data["total_pred"]
            
            accuracy = (tp / total_gt * 100) if total_gt > 0 else 0
            precision = (tp / total_pred * 100) if total_pred > 0 else 0
            recall = (tp / total_gt * 100) if total_gt > 0 else 0
            f1 = (2 * precision * recall / (precision + recall)) if (precision + recall) > 0 else 0
            
            w.writerow({
                "model": model_name,
                "class": class_name,
                "total_gt": total_gt,
                "total_pred": total_pred,
                "tp": tp,
                "fp": fp,
                "fn": fn,
                "accuracy": f"{accuracy:.1f}",
                "precision": f"{precision:.1f}",
                "recall": f"{recall:.1f}",
                "f1_score": f"{f1:.1f}"
            })
    
    print(f"\n✅ Per-class metrics saved to: {per_class_csv}")

    print("\n" + "="*80)
    print("EVALUATION COMPLETE")
    print("="*80)

if __name__ == "__main__":
    main()