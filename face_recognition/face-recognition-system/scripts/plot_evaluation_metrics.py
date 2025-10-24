import argparse
import csv
from pathlib import Path
import math
from collections import defaultdict, Counter
import numpy as np
import matplotlib.pyplot as plt
import glob
import sys

def safe_float(x):
    try:
        return float(x)
    except Exception:
        return float("nan")

def load_results_from_csv(path, model_override=None):
    rows = []
    with open(path, newline="", encoding="utf-8") as f:
        rdr = csv.DictReader(f)
        for r in rdr:
            row = {k: (v if v is not None else "") for k, v in r.items()}
            # normalize field names
            model = row.get("model") or model_override or row.get("Model") or ""
            status = row.get("status") or row.get("Status") or ""
            predicted = row.get("predicted") or row.get("pred") or row.get("prediction") or ""
            confidence = row.get("confidence") or row.get("Confidence") or ""
            time_s = row.get("time") or row.get("Time") or ""
            rows.append({
                "file": row.get("file",""),
                "model": model,
                "status": status.lower(),
                "predicted": predicted,
                "confidence": safe_float(confidence),
                "time": safe_float(time_s)
            })
    return rows

def aggregate(rows):
    per = defaultdict(lambda: {"rows": [], "total":0, "recognized":0, "times":[], "confs":[], "names": Counter()})
    for r in rows:
        m = r["model"] or "Unknown"
        per[m]["rows"].append(r)
        per[m]["total"] += 1
        if r["status"] == "ok" and r["predicted"] and r["predicted"] != "Unknown":
            per[m]["recognized"] += 1
            if not math.isnan(r["confidence"]):
                per[m]["confs"].append(r["confidence"])
            per[m]["names"][r["predicted"]] += 1
        if not math.isnan(r["time"]):
            per[m]["times"].append(r["time"])
    return per

def plot_summary(per, out_dir, top_n_names=10):
    out_dir.mkdir(parents=True, exist_ok=True)
    models = list(per.keys())
    processed = [per[m]["total"] for m in models]
    rec_rates = [ (per[m]["recognized"] / per[m]["total"] * 100.0) if per[m]["total"] else 0.0 for m in models]
    avg_times = [ (sum(per[m]["times"]) / len(per[m]["times"])) if per[m]["times"] else 0.0 for m in models]
    avg_confs = [ (sum(per[m]["confs"]) / len(per[m]["confs"])) if per[m]["confs"] else 0.0 for m in models]

    # bar: processed count
    plt.figure(figsize=(8,4))
    plt.bar(models, processed, color="#4c72b0")
    plt.ylabel("Processed")
    plt.title("Images processed per model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "processed_per_model.png")
    plt.close()

    # bar: recognition rate
    plt.figure(figsize=(8,4))
    plt.bar(models, rec_rates, color="#55a868")
    plt.ylabel("Recognition rate (%)")
    plt.title("Recognition rate per model")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "recognition_rate_per_model.png")
    plt.close()

    # bar: avg time
    plt.figure(figsize=(8,4))
    plt.bar(models, avg_times, color="#c44e52")
    plt.ylabel("Avg time (s)")
    plt.title("Average processing time per image")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_time_per_model.png")
    plt.close()

    # bar: avg confidence
    plt.figure(figsize=(8,4))
    plt.bar(models, avg_confs, color="#8172b3")
    plt.ylabel("Avg confidence (recognized)")
    plt.title("Average confidence per model (recognized only)")
    plt.xticks(rotation=45, ha="right")
    plt.tight_layout()
    plt.savefig(out_dir / "avg_confidence_per_model.png")
    plt.close()

    # cumulative recognition curves (overlay)
    plt.figure(figsize=(8,4))
    for m in models:
        rows = per[m]["rows"]
        # compute cumulative recognized fraction in input order
        flags = [1 if (r["status"]=="ok" and r["predicted"] and r["predicted"]!="Unknown") else 0 for r in rows]
        if not flags:
            continue
        idx = np.arange(1, len(flags)+1)
        cum = np.cumsum(flags) / idx
        plt.plot(idx, cum, marker=".", linewidth=1, label=m)
    plt.xlabel("Images processed")
    plt.ylabel("Cumulative recognition rate")
    plt.title("Cumulative recognition rate (per model)")
    plt.legend()
    plt.grid(alpha=0.3)
    plt.tight_layout()
    plt.savefig(out_dir / "cumulative_recognition_all_models.png")
    plt.close()

    # per-model time histograms + confidence histograms + top names
    for m in models:
        times = per[m]["times"]
        confs = per[m]["confs"]
        top = per[m]["names"].most_common(top_n_names)

        if times:
            plt.figure(figsize=(6,4))
            plt.hist(times, bins=30, color="#4c72b0", edgecolor="k", alpha=0.8)
            plt.xlabel("Time (s)")
            plt.ylabel("Count")
            plt.title(f"{m} per-image processing time")
            plt.tight_layout()
            plt.savefig(out_dir / f"{m}_time_histogram.png")
            plt.close()

        if confs:
            plt.figure(figsize=(6,4))
            plt.hist(confs, bins=20, color="#55a868", edgecolor="k", alpha=0.8)
            plt.xlabel("Confidence")
            plt.ylabel("Count")
            plt.title(f"{m} confidence distribution (recognized)")
            plt.tight_layout()
            plt.savefig(out_dir / f"{m}_confidence_histogram.png")
            plt.close()

        if top:
            names, counts = zip(*top)
            plt.figure(figsize=(8, max(3, len(names)*0.4)))
            plt.barh(range(len(names))[::-1], counts, color="#c44e52")
            plt.yticks(range(len(names))[::-1], names)
            plt.xlabel("Predictions")
            plt.title(f"{m} top predicted names")
            plt.tight_layout()
            plt.savefig(out_dir / f"{m}_top_names.png")
            plt.close()

    print("Saved charts to:", out_dir)

def main():
    ap = argparse.ArgumentParser(description="Plot evaluation metrics from CSV logs")
    ap.add_argument(
        "--logs-dir",
        default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\face_recognition\face-recognition-system\data\logs",
        help="Folder containing evaluation CSV logs"
    )
    ap.add_argument("--out", default=None, help="Output folder for charts (defaults to logs-dir)")
    ap.add_argument("--top-n-names", type=int, default=10)
    args = ap.parse_args()

    logs_dir = Path(args.logs_dir)
    if not logs_dir.exists():
        print("Logs folder not found:", logs_dir)
        sys.exit(1)
    base_out = Path(args.out) if args.out else logs_dir
    out_dir = base_out / "metrics"
    out_dir.mkdir(parents=True, exist_ok=True)

    rows = []
    # model comparison results
    for p in sorted(glob.glob(str(logs_dir / "model_comparison_results_*.csv"))):
        rows.extend(load_results_from_csv(p))
    # dlib-only results
    for p in sorted(glob.glob(str(logs_dir / "dlib_results_*.csv"))):
        rows.extend(load_results_from_csv(p, model_override="Dlib"))

    if not rows:
        print("No CSV results found in", logs_dir)
        sys.exit(1)

    per = aggregate(rows)
    plot_summary(per, out_dir, top_n_names=args.top_n_names)

if __name__ == "__main__":
    main()