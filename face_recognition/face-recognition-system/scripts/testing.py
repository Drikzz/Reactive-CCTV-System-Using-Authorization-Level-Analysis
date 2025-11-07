import os
import sys
import time
import csv
import argparse
from pathlib import Path
from datetime import datetime
import importlib.util
import concurrent.futures
import cv2
import numpy as np

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

def build_ground_truth_from_dataset(dataset_path):
    """
    Build ground truth mapping from LevelsAuthorization dataset structure.
    Returns tuple: (gt_map, class_names)
      - gt_map: dict mapping image filename or stem -> class/person name
      - class_names: set of top-level class/folder names found in dataset (e.g. "Authorized", "Partially Authorized")
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
    
    # Iterate through dataset to build file-level mapping if present
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
                            # Map both full filename and stem
                            gt_map[img_file.name] = person_name
                            gt_map[img_file.stem] = person_name
                            image_count += 1
        else:
            # Flat structure - images directly in person/class folder
            for img_file in person_dir.glob("*"):
                if img_file.suffix.lower() in ['.jpg', '.jpeg', '.png']:
                    gt_map[img_file.name] = person_name
                    gt_map[img_file.stem] = person_name
                    image_count += 1
    
    print(f"[INFO] Ground truth built: {person_count} persons/classes, {image_count} images (file-level mappings)")
    print(f"[INFO] Class names found: {sorted(class_names)}")
    return gt_map, class_names

def match_crop_to_ground_truth(crop_filename, gt_map, class_names=None):
    """
    Match a face crop filename to ground truth.
    Strategies (in order):
      1. Exact filename match against gt_map
      2. Stem match against gt_map
     
      4. Substring match of class names (class_names)
      5. Fuzzy-like match: remove spaces/punctuation and check inclusion
    Returns matched class/person name or None.
    """
    crop_path = Path(crop_filename)
    stem = crop_path.stem
    name = crop_path.name
    
    # Strategy 1: Direct match (file name)
    if name in gt_map:
        return gt_map[name]
    
    # Strategy 2: Stem match
    if stem in gt_map:
        return gt_map[stem]
    
    # Strategy 3: Parse crop filename: common patterns "Person_frameXXX", "Class_123"
    if '_' in stem:
        first_part = stem.split('_')[0]
        # Check direct equality against gt_map values
        for v in set(gt_map.values()):
            if first_part.lower() == v.lower():
                return v
        # Check against class_names
        if class_names:
            for cls in class_names:
                if first_part.lower() == cls.lower():
                    return cls
    
    # Strategy 4: Substring match against class_names
    if class_names:
        stem_lower = stem.lower()
        for cls in sorted(class_names, key=lambda s: -len(s)):
            cls_clean = "".join(c.lower() for c in cls if c.isalnum())
            if cls.lower() in stem_lower or cls_clean in stem_lower:
                return cls
    
    # Strategy 5: Fuzzy-like match - check if any class name tokens appear in filename
    if class_names:
        stem_alnum = "".join(c.lower() for c in stem if c.isalnum())
        for cls in class_names:
            cls_alnum = "".join(c.lower() for c in cls if c.isalnum())
            if cls_alnum and cls_alnum in stem_alnum:
                return cls
    
    return None

def ensure_models_trained():
    """Ensure models are trained with LevelsAuthorization dataset before evaluation"""
    dataset_path = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\datasets\LevelsAuthorization"
    
    print("Checking if models are trained with LevelsAuthorization dataset...")
    
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

def _norm(s):
    """Normalize labels for robust comparison: keep only alphanumeric, lowercase."""
    if s is None:
        return ""
    s = str(s)
    return "".join(c.lower() for c in s if c.isalnum())

def main():
    # Add this check at the beginning of main()
    if not ensure_models_trained():
        print("ERROR: Models not trained. Please train models with LevelsAuthorization dataset first.")
        print("Run the training scripts in face_recognition/Facenet/, face_recognition/ArcFace/, etc.")
        return
    
    parser = argparse.ArgumentParser(description="Evaluate face recognition against LevelsAuthorization ground truth")
    parser.add_argument("--crops", 
                       default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\face_recognition\face-recognition-system\data\face_crops",
                       help="Folder with face crop images")
    parser.add_argument("--dataset",
                       default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\datasets\LevelsAuthorization",
                       help="Path to LevelsAuthorization dataset for ground truth")
    parser.add_argument("--out", 
                       default=str(Path(__file__).resolve().parents[1] / "data" / "logs"),
                       help="Output folder for CSV logs")
    parser.add_argument("--facenet", help="Path to facenet_main.py (if omitted, auto-locate)", default=None)
    parser.add_argument("--arcface", help="Path to arcface_main.py (if omitted, auto-locate)", default=None)
    parser.add_argument("--limit", type=int, default=0, help="Limit number of images (0 = all)")
    parser.add_argument("--workers", type=int, default=4, help="Number of worker processes for evaluation")
    parser.add_argument("--use-gpu", action="store_true", help="Attempt to run recognizers on GPU")
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    dataset_dir = Path(args.dataset)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    # Build ground truth from LevelsAuthorization dataset
    print("\n" + "="*80)
    print("BUILDING GROUND TRUTH FROM DATASET")
    print("="*80)
    gt_map, class_names = build_ground_truth_from_dataset(dataset_dir)
    
    if not gt_map and not class_names:
        print("[ERROR] No ground truth data found. Check dataset path.")
        return
    
    if gt_map:
        persons_list = sorted(set(gt_map.values()))
    else:
        persons_list = sorted(class_names)
    print(f"\n[INFO] Ground truth persons/classes: {persons_list}")

    if not crops_dir.exists():
        print(f"[ERROR] Crops folder not found: {crops_dir}")
        return

    # autolocate modules if not provided
    repo_root = Path(__file__).resolve().parents[3]
    if args.facenet:
        facenet_path = Path(args.facenet)
    else:
        facenet_path = repo_root / "face_recognition" / "Facenet" / "facenet_main.py"
    if args.arcface:
        arcface_path = Path(args.arcface)
    else:
        arcface_path = repo_root / "face_recognition" / "ArcFace" / "arcface_main.py"

    print("\n" + "="*80)
    print("LOADING RECOGNITION MODULES")
    print("="*80)
    print(f" FaceNet: {facenet_path}")
    print(f" ArcFace: {arcface_path}")

    # If user requested GPU, export an env flag
    if args.use_gpu:
        os.environ["FORCE_GPU"] = "1"
        try:
            import torch
            if torch.cuda.is_available():
                print(f"[INFO] CUDA available: {torch.cuda.get_device_name(0)}")
            else:
                print("[WARN] --use-gpu requested but CUDA not available")
        except Exception:
            print("[WARN] PyTorch not available in main process")

    # load modules
    facenet_mod = None
    arcface_mod = None

    try:
        facenet_mod = load_module_from_path("facenet_main", facenet_path)
        if args.use_gpu and facenet_mod is not None:
            try:
                import torch
                if hasattr(facenet_mod, "set_device"):
                    facenet_mod.set_device("cuda")
                setattr(facenet_mod, "DEVICE", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            except Exception:
                pass
        print("✅ FaceNet module loaded")
    except Exception as e:
        print(f"❌ FaceNet module load failed: {e}")

    try:
        arcface_mod = load_module_from_path("arcface_main", arcface_path)
        if args.use_gpu and arcface_mod is not None:
            try:
                import torch
                if hasattr(arcface_mod, "set_device"):
                    arcface_mod.set_device("cuda")
                setattr(arcface_mod, "DEVICE", torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu"))
            except Exception:
                pass
        print("✅ ArcFace module loaded")
    except Exception as e:
        print(f"❌ ArcFace module load failed: {e}")

    # Prepare ArcFaceSystem instance if available
    arcface_system = None
    if arcface_mod and hasattr(arcface_mod, "ArcFaceSystem"):
        try:
            if args.use_gpu:
                try:
                    arcface_system = arcface_mod.ArcFaceSystem(use_yolo=False, device="cuda")
                except TypeError:
                    arcface_system = arcface_mod.ArcFaceSystem(use_yolo=False)
            else:
                arcface_system = arcface_mod.ArcFaceSystem(use_yolo=False)
            print("✅ ArcFaceSystem initialized")
        except Exception as e:
            print(f"❌ ArcFaceSystem init failed: {e}")

    # Prepare predictor callables
    predictors = []

    if facenet_mod and hasattr(facenet_mod, "recognize_face_in_crop"):
        def facenet_wrapper(img):
            h, w = img.shape[:2]
            ok, res, dt, err = safe_call(facenet_mod.recognize_face_in_crop, img, img, (0,0,w,h))
            if not ok:
                return {"status":"error","error":err}
            return {"status":"ok","result":res,"time":dt}
        predictors.append(("FaceNet", facenet_wrapper))
    else:
        print("[WARN] FaceNet function not available; skipping")

    if arcface_system and hasattr(arcface_system, "recognize_face_in_crop_enhanced"):
        def arcface_wrapper(img):
            h,w = img.shape[:2]
            ok, res, dt, err = safe_call(arcface_system.recognize_face_in_crop_enhanced, img, img, (0,0,w,h), 0)
            if not ok:
                return {"status":"error","error":err}
            return {"status":"ok","result":res,"time":dt}
        predictors.append(("ArcFace", arcface_wrapper))
    else:
        print("[WARN] ArcFace recognizer not available; skipping")

    if not predictors:
        print("[ERROR] No recognizers available. Exiting.")
        return

    # Get image paths
    img_paths = sorted([p for p in crops_dir.glob("*") if p.suffix.lower() in [".jpg",".jpeg",".png"]])
    if args.limit:
        img_paths = img_paths[:args.limit]

    print("\n" + "="*80)
    print(f"PROCESSING {len(img_paths)} IMAGES")
    print("="*80)

    results = []
    model_stats = {name: {
        "total":0, 
        "recognized":0, 
        "times":[],
        "confs":[],
        "names":{}, 
        "correct": 0,
        "has_gt": 0,
        "correct_with_gt": 0,
        # NEW: Per-class statistics
        "class_stats": {}  # class_name -> {tp, fp, fn, tn}
    } for name, _ in predictors}

    workers = min(args.workers, max(1, os.cpu_count() or 1))
    print(f"Using {workers} worker processes...\n")

    with concurrent.futures.ProcessPoolExecutor(max_workers=workers) as exe:
        facenet_path_str = str(facenet_path) if facenet_mod else ""
        arcface_path_str = str(arcface_path) if arcface_mod else ""
        
        futures = {
            exe.submit(_process_image_worker, str(p), facenet_path_str, arcface_path_str, dict(gt_map)): p 
            for p in img_paths
        }
        
        completed = 0
        for fut in concurrent.futures.as_completed(futures):
            try:
                rows = fut.result()
            except Exception as e:
                print(f"[ERROR] Worker failed: {e}")
                continue
            
            completed += 1
            if completed % 10 == 0:
                print(f"[PROGRESS] Processed {completed}/{len(img_paths)} images...")
            
            for row in rows:
                # Get ground truth using matching function
                gt = match_crop_to_ground_truth(row["file"], gt_map, class_names)
                row["ground_truth"] = gt or "Unknown"

                # Update stats
                mn = row["model"]
                model_stats[mn]["total"] += 1
                
                if row["status"] == "ok" and row["predicted"] and row["predicted"] != "Unknown":
                    model_stats[mn]["recognized"] += 1
                    model_stats[mn]["confs"].append(float(row.get("confidence", 0.0) or 0.0))
                    model_stats[mn]["names"].setdefault(row["predicted"], 0)
                    model_stats[mn]["names"][row["predicted"]] += 1
                
                # Track accuracy
                if gt and gt != "Unknown":
                    model_stats[mn]["has_gt"] += 1
                    if row["status"] == "ok" and _norm(row.get("predicted")) == _norm(gt):
                        model_stats[mn]["correct"] += 1
                        model_stats[mn]["correct_with_gt"] += 1
                
                # NEW: Track per-class statistics (for confusion matrix)
                pred = _norm(row.get("predicted", ""))
                gt_norm = _norm(gt) if gt else ""
                
                if gt and gt != "Unknown":
                    # Initialize class stats if needed
                    if gt not in model_stats[mn]["class_stats"]:
                        model_stats[mn]["class_stats"][gt] = {
                            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                            "total_gt": 0, "total_pred": 0
                        }
                    
                    model_stats[mn]["class_stats"][gt]["total_gt"] += 1
                    
                    # True Positive: predicted correctly
                    if pred == gt_norm:
                        model_stats[mn]["class_stats"][gt]["tp"] += 1
                    else:
                        # False Negative: should be this class but predicted something else
                        model_stats[mn]["class_stats"][gt]["fn"] += 1
                
                # Track false positives
                if row["status"] == "ok" and row.get("predicted") and row["predicted"] != "Unknown":
                    pred_class = row["predicted"]
                    if pred_class not in model_stats[mn]["class_stats"]:
                        model_stats[mn]["class_stats"][pred_class] = {
                            "tp": 0, "fp": 0, "fn": 0, "tn": 0,
                            "total_gt": 0, "total_pred": 0
                        }
                    
                    model_stats[mn]["class_stats"][pred_class]["total_pred"] += 1
                    
                    # If prediction doesn't match ground truth, it's a false positive
                    if gt and gt != "Unknown" and _norm(pred_class) != gt_norm:
                        model_stats[mn]["class_stats"][pred_class]["fp"] += 1
                
                try:
                    model_stats[mn]["times"].append(float(row.get("time", 0.0) or 0.0))
                except Exception:
                    pass
                
                results.append(row)

    # Save detailed CSV
    print("\n" + "="*80)
    print("SAVING RESULTS")
    print("="*80)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    out_csv = out_dir / f"evaluation_results_{timestamp}.csv"
    keys = ["file","model","status","predicted","confidence","time","ground_truth","error","match"]
    
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

    print(f"✅ Results saved to: {out_csv}")

    # Print summary
    print("\n" + "="*80)
    print("EVALUATION SUMMARY")
    print("="*80)
    
    for model_name, stats in model_stats.items():
        total = stats["total"]
        recognized = stats["recognized"]
        has_gt = stats["has_gt"]
        correct_with_gt = stats["correct_with_gt"]
        
        avg_time = (sum(stats["times"]) / len(stats["times"])) if stats["times"] else 0.0
        avg_conf = (sum(stats["confs"]) / len(stats["confs"])) if stats["confs"] else 0.0
        most_common = sorted(stats["names"].items(), key=lambda x: x[1], reverse=True)[:5]
        
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
        print(f"  Top 5 predictions: {most_common}")
        
        # NEW: Per-class metrics
        if stats["class_stats"]:
            print(f"\n  {'─'*38}")
            print(f"  PER-CLASS METRICS:")
            print(f"  {'─'*38}")
            
            # Sort classes for consistent display
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
                
                # Calculate per-class metrics
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
    per_class_csv = out_dir / f"per_class_metrics_{timestamp}.csv"
    with open(per_class_csv, "w", newline="", encoding="utf-8") as f:
        fieldnames = ["model", "class", "total_gt", "total_pred", "tp", "fp", "fn", 
                     "accuracy", "precision", "recall", "f1_score"]
        w = csv.DictWriter(f, fieldnames=fieldnames)
        w.writeheader()
        
        for model_name, stats in model_stats.items():
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

def _process_image_worker(img_path_str, facenet_path_str, arcface_path_str, gt_map_dict):
    """Worker process for parallel image evaluation"""
    from pathlib import Path
    import importlib.util
    import cv2
    import time
    import os

    img_p = Path(img_path_str)
    img = cv2.imread(str(img_p))
    if img is None:
        return []

    # Per-process module cache
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
        
        # GPU hints
        try:
            if os.getenv("FORCE_GPU") == "1":
                try:
                    import torch
                    if hasattr(mod, "set_device"):
                        try:
                            mod.set_device("cuda")
                        except Exception:
                            pass
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

    rows = []
    
    def _safe_call_local(func, *a, **kw):
        t0 = time.time()
        try:
            res = func(*a, **kw)
            return True, res, time.time() - t0, None
        except Exception as e:
            return False, None, time.time() - t0, str(e)

    # Run FaceNet
    if facenet_mod and hasattr(facenet_mod, "recognize_face_in_crop"):
        ok, res, dt, err = _safe_call_local(facenet_mod.recognize_face_in_crop, img, img, (0,0,img.shape[1], img.shape[0]))
        if not ok:
            rows.append({
                "file": img_p.name, "model": "FaceNet", "status":"error", "error": err,
                "predicted":"", "confidence":"", "time": dt
            })
        else:
            if isinstance(res, dict):
                pred = res.get("name") or res.get("recognized_name") or res.get("predicted","Unknown")
                conf = res.get("confidence") or 0.0
            else:
                pred, conf = (str(res), 0.0)
            rows.append({
                "file": img_p.name, "model": "FaceNet", "status":"ok", "error":[],
                "predicted": pred, "confidence": float(conf), "time": float(dt)
            })

    # Run ArcFace
    if arcface_mod and hasattr(arcface_mod, "ArcFaceSystem"):
        try:
            if not hasattr(_process_image_worker, "_arc_sys"):
                try:
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
                    rows.append({
                        "file": img_p.name, "model": "ArcFace", "status":"error", "error": err,
                        "predicted":"", "confidence":"", "time": dt
                    })
                else:
                    if isinstance(res, dict):
                        pred = res.get("name") or res.get("recognized_name") or res.get("predicted","Unknown")
                        conf = res.get("confidence") or 0.0
                    else:
                        pred, conf = (str(res), 0.0)
                    rows.append({
                        "file": img_p.name, "model": "ArcFace", "status":"ok", "error":"",
                        "predicted": pred, "confidence": float(conf), "time": float(dt)
                    })
        except Exception as e:
            rows.append({
                "file": img_p.name, "model": "ArcFace", "status":"error", "error": str(e),
                "predicted":"", "confidence":"", "time":0.0
            })

    return rows

if __name__ == "__main__":
    main()