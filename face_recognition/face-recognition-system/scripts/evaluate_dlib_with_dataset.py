import csv, time, importlib.util, argparse, concurrent.futures
from pathlib import Path
from collections import Counter
import numpy as np
import cv2

ROOT = Path(r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis")
DEFAULT_CROPS = ROOT / "face_recognition" / "face-recognition-system" / "data" / "face_crops"
DEFAULT_DLIB = ROOT / "face_recognition" / "Dlibs CNN" / "dilib_cnn_main_optimized.py"
DEFAULT_LOGS = ROOT / "face_recognition" / "face-recognition-system" / "data" / "logs"
DEFAULT_DATASET = ROOT / "datasets" / "faces"

def load_module(path):
    spec = importlib.util.spec_from_file_location("dlib_mod", str(path))
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod

def _norm(s):
    if not s:
        return ""
    return "".join(ch.lower() for ch in str(s) if ch.isalnum())

def load_gt_map(crops_dir):
    gt_map = {}
    gt_csv = Path(crops_dir) / "ground_truth.csv"
    if not gt_csv.exists():
        return gt_map
    try:
        with gt_csv.open(encoding="utf-8", newline="") as fh:
            # try DictReader first (handles header row like "file,name")
            try:
                fh.seek(0)
                dr = csv.DictReader(fh)
                if dr.fieldnames and any('file' == f.lower() for f in dr.fieldnames):
                    name_field = None
                    for f in dr.fieldnames:
                        if f and f.lower() in ("name", "ground_truth", "gt", "label"):
                            name_field = f
                            break
                    if name_field:
                        for r in dr:
                            fn = (r.get('file') or "").strip()
                            name = (r.get(name_field) or "").strip()
                            if fn:
                                gt_map[fn] = name
                                gt_map[Path(fn).stem] = name
            except Exception:
                pass
            # fallback to simple CSV rows
            fh.seek(0)
            rdr = csv.reader(fh)
            for r in rdr:
                if not r:
                    continue
                if len(r) >= 2:
                    fn = r[0].strip(); name = r[1].strip()
                    if fn:
                        gt_map[fn] = name
                        gt_map[Path(fn).stem] = name
    except Exception:
        # ignore parse errors and return whatever was found
        pass
    return gt_map

def load_dataset_embeddings(dataset_root):
    emb_p = Path(dataset_root) / "embeddings.npy"
    names_p = Path(dataset_root) / "names.npy"
    if emb_p.exists() and names_p.exists():
        embs = np.load(str(emb_p)).astype(np.float32)
        names = [str(x) for x in np.load(str(names_p)).astype(str)]
        embs = embs / (np.linalg.norm(embs, axis=1, keepdims=True) + 1e-10)
        return embs, names
    return None, None

def find_nearest(emb, dataset_embs, dataset_names, thr=0.85):
    if emb is None or dataset_embs is None:
        return "", 0.0
    try:
        e = np.asarray(emb, dtype=np.float32).reshape(-1)
        e = e / (np.linalg.norm(e) + 1e-10)
        sims = dataset_embs.dot(e)
        idx = int(np.argmax(sims))
        score = float(sims[idx])
        if score >= thr:
            return dataset_names[idx], score
        return "", score
    except Exception:
        return "", 0.0

def call_recognize(dlib_mod, img):
    """
    Call recognizer function from the dlib module. Heuristics to find function name and returned structure.
    Expected to return dict or string; normalized to dict below.
    """
    try:
        func = getattr(dlib_mod, "recognize_face_in_crop_optimized", None)
        if func is None:
            for n in dir(dlib_mod):
                if "recognize" in n.lower() and callable(getattr(dlib_mod, n)):
                    func = getattr(dlib_mod, n)
                    break
        if func is None:
            return {"status": "error", "error": "no_recognize_func"}

        t0 = time.time()
        try:
            # Some dlib wrappers accept (img, original_img, bbox) or just (img,)
            # Try common signatures
            try:
                res = func(img, img, (0, 0, img.shape[1], img.shape[0]))
            except TypeError:
                try:
                    res = func(img)
                except TypeError:
                    res = func(img, (0, 0, img.shape[1], img.shape[0]))
        except Exception as e:
            return {"status": "error", "error": str(e)}
        dur = time.time() - t0

        if isinstance(res, dict):
            pred = res.get("name") or res.get("recognized_name") or res.get("predicted") or "Unknown"
            conf = float(res.get("confidence", 0.0) or 0.0)
            emb = res.get("embedding") or res.get("vector") or None
            return {"status": "ok", "predicted": pred, "confidence": conf, "time": dur, "embedding": emb}
        else:
            # If recognizer returns a simple string / label
            return {"status": "ok", "predicted": str(res) or "Unknown", "confidence": 0.0, "time": dur, "embedding": None}
    except Exception as e:
        return {"status": "error", "error": str(e)}

def _process_one(img_path_str, dlib_path_str, dataset_embs_present):
    p = Path(img_path_str)
    img = cv2.imread(str(p))
    if img is None:
        return {"file": p.name, "status": "error", "error": "read_failed", "predicted": "", "confidence": 0.0, "time": 0.0, "embedding": None}
    # load module cached per worker
    if not hasattr(_process_one, "_cache"):
        _process_one._cache = {}
    key = dlib_path_str
    if key not in _process_one._cache:
        try:
            _process_one._cache[key] = load_module(Path(dlib_path_str))
        except Exception as e:
            return {"file": p.name, "status": "error", "error": f"load_failed:{e}", "predicted": "", "confidence": 0.0, "time": 0.0, "embedding": None}
    dlib_mod = _process_one._cache[key]
    out = call_recognize(dlib_mod, img)
    out["file"] = p.name
    # if module didn't return embedding but dataset verification requested, try compute_embedding_for_crop if present
    if out.get("status") == "ok" and out.get("embedding") is None and dataset_embs_present:
        emb_fn = getattr(dlib_mod, "compute_embedding_for_crop", None)
        if emb_fn:
            try:
                emb = emb_fn(img)
                out["embedding"] = emb
            except Exception:
                out["embedding"] = None
    return out

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--crops", default=str(DEFAULT_CROPS))
    parser.add_argument("--dlib", default=str(DEFAULT_DLIB))
    parser.add_argument("--out", default=str(DEFAULT_LOGS))
    parser.add_argument("--dataset", default=str(DEFAULT_DATASET))
    parser.add_argument("--workers", type=int, default=4)
    parser.add_argument("--match-threshold", type=float, default=0.85)
    parser.add_argument("--limit", type=int, default=0)
    args = parser.parse_args()

    crops_dir = Path(args.crops)
    out_dir = Path(args.out); out_dir.mkdir(parents=True, exist_ok=True)
    if not Path(args.dlib).exists():
        print("Dlib module not found:", args.dlib); return
    if not crops_dir.exists():
        print("Crops dir not found:", crops_dir); return

    gt_map = load_gt_map(crops_dir)
    dataset_embs, dataset_names = load_dataset_embeddings(args.dataset)
    dataset_present = dataset_embs is not None and dataset_names is not None

    img_paths = sorted([p for p in crops_dir.glob("*") if p.suffix.lower() in (".jpg", ".png")])
    if args.limit and args.limit > 0:
        img_paths = img_paths[:args.limit]
    total = len(img_paths)
    if total == 0:
        print("No crop images found"); return

    print(f"Running Dlib evaluator on {total} images (workers={args.workers})")
    results = []
    stats = {"total": 0, "recognized": 0, "times": [], "confs": [], "names": Counter(), "correct": 0}
    with concurrent.futures.ProcessPoolExecutor(max_workers=args.workers) as exe:
        futures = {exe.submit(_process_one, str(p), str(args.dlib), dataset_present): p for p in img_paths}
        for fut in concurrent.futures.as_completed(futures):
            try:
                row = fut.result()
            except Exception as e:
                row = {"file": str(futures[fut].name), "status": "error", "error": str(e), "predicted": "", "confidence": 0.0, "time": 0.0, "embedding": None}
            fn = row.get("file", "")
            stats["total"] += 1
            if row.get("status") == "ok":
                pred = row.get("predicted", "") or ""
                conf = float(row.get("confidence", 0.0) or 0.0)
                t = float(row.get("time", 0.0) or 0.0)
                stats["times"].append(t)
                if pred and pred != "Unknown":
                    stats["recognized"] += 1
                    stats["confs"].append(conf)
                    stats["names"][pred] += 1
                # ground truth resolution (match evaluate_models_on_crops.py logic)
                gt = gt_map.get(fn) or gt_map.get(Path(fn).stem)
                ds_pred = ""
                ds_conf = ""
                if row.get("embedding") is not None and dataset_present:
                    ds_pred, ds_conf = find_nearest(row.get("embedding"), dataset_embs, dataset_names, thr=args.match_threshold)
                # if no explicit GT, fall back to dataset-based assignment
                gt_final = gt or ds_pred or ""
                if gt_final and pred and _norm(pred) == _norm(gt_final):
                    stats["correct"] += 1
                # append row
                out_row = {
                    "file": fn, "status": row.get("status"), "predicted": pred,
                    "confidence": conf, "time": t, "ground_truth": gt_final,
                    "dataset_predicted": ds_pred, "dataset_confidence": f"{ds_conf:.4f}" if ds_conf else ""
                }
            else:
                out_row = {"file": fn, "status": row.get("status"), "predicted": "", "confidence": "", "time": row.get("time", 0.0), "ground_truth": "", "dataset_predicted": "", "dataset_confidence": ""}
            results.append(out_row)

    # save CSV
    ts = int(time.time())
    out_csv = out_dir / f"dlib_model_comparison_{ts}.csv"
    keys = ["file", "status", "predicted", "confidence", "time", "ground_truth", "dataset_predicted", "dataset_confidence"]
    with out_csv.open("w", newline="", encoding="utf-8") as fh:
        w = csv.DictWriter(fh, fieldnames=keys)
        w.writeheader()
        for r in results:
            w.writerow({k: r.get(k, "") for k in keys})

    # --- compute accuracy over rows that have explicit ground-truth and dataset-based agreement
    rows_with_gt = [r for r in results if (r.get("ground_truth") or "").strip() != ""]
    with_gt = len(rows_with_gt)
    correct_with_gt = sum(1 for r in rows_with_gt if _norm(r.get("predicted", "")) == _norm(r.get("ground_truth", "")))

    rows_with_ds = [r for r in results if (r.get("dataset_predicted") or "").strip() != ""]
    with_ds = len(rows_with_ds)
    correct_vs_dataset = sum(1 for r in rows_with_ds if _norm(r.get("predicted", "")) == _norm(r.get("dataset_predicted", "")))

    # print summary
    print("\n=== EVALUATION SUMMARY ===\n")
    avg_time = (sum(stats["times"]) / len(stats["times"])) if stats["times"] else 0.0
    avg_conf = (sum(stats["confs"]) / len(stats["confs"])) if stats["confs"] else 0.0
    print("Model: Dlib")
    print(f"  Processed: {stats['total']}")
    print(f"  Recognized (not 'Unknown'): {stats['recognized']} ({(stats['recognized'] / stats['total'] * 100) if stats['total'] else 0:.1f}%)")
    print(f"  Correct (predicted == ground_truth): {stats['correct']} ({(stats['correct'] / stats['total'] * 100) if stats['total'] else 0:.1f}%)")
    # prediction-accuracy = how correct the predictions are (correct / recognized)
    if stats["recognized"]:
        pred_acc = stats["correct"] / stats["recognized"]
        print(f"  Prediction accuracy (correct / recognized): {stats['correct']}/{stats['recognized']} ({pred_acc*100:.1f}%)")
    else:
        print("  Prediction accuracy (correct / recognized): no predictions made")
    # accuracy over GT and F1
    if with_gt:
        print(f"  Accuracy over rows with GT: {correct_with_gt}/{with_gt} ({(correct_with_gt / with_gt * 100):.1f}%)")
    else:
        print("  Accuracy over rows with GT: no explicit GT rows")
    if with_ds:
        print(f"  Dataset-based agreement: {correct_vs_dataset}/{with_ds} ({(correct_vs_dataset / with_ds * 100):.1f}%)")
    else:
        print("  Dataset-based agreement: no dataset matches computed")
    # compute precision/recall/F1 where possible
    try:
        precision = (stats["correct"] / stats["recognized"]) if stats["recognized"] else float("nan")
    except Exception:
        precision = float("nan")
    try:
        recall = (correct_with_gt / with_gt) if with_gt else float("nan")
    except Exception:
        recall = float("nan")
    f1 = (2 * precision * recall / (precision + recall)) if (precision and recall and not (precision != precision or recall != recall)) else float("nan")
    if not (precision != precision):
        print(f"  Precision (correct/recognized): {precision*100:.1f}%")
    if not (recall != recall):
        print(f"  Recall (correct/with_GT): {recall*100:.1f}%")
    if not (f1 != f1):
        print(f"  F1 score: {f1:.3f}")
    print(f"  Avg time / image: {avg_time:.4f}s")
    print(f"  Avg confidence (recognized): {avg_conf:.3f}")
    print(f"  Top predicted names: {stats['names'].most_common(10)}")
    print(f"\nDetailed results saved to: {out_csv}")

if __name__ == "__main__":
    main()