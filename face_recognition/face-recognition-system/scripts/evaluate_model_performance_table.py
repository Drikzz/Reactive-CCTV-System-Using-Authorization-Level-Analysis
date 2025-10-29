import math
from pathlib import Path
import csv
from collections import defaultdict
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np

# Path to your CSV logs folder
LOGS = Path("face_recognition/face-recognition-system/data/logs")

def norm(s):
    return "".join(c.lower() for c in (s or "") if c.isalnum())

def latest_csvs():
    files = sorted(LOGS.glob("*.csv"), key=lambda p: p.stat().st_mtime, reverse=True)
    return files

def collect_stats(files):
    stats = defaultdict(lambda: {"processed": 0, "recognized": 0, "tp": 0, "fp": 0, "with_gt": 0})
    for f in files:
        with f.open(encoding="utf-8", errors="ignore") as fh:
            rdr = csv.DictReader(fh)
            for r in rdr:
                model = (r.get("model") or "").strip() or None
                if model is None or model == "":
                    name = f.name.lower()
                    if "facenet" in name:
                        model = "FaceNet"
                    elif "arcface" in name or "insight" in name:
                        model = "ArcFace (InsightFace)"
                    elif "dlib" in name:
                        model = "Dlib CNN"
                    else:
                        model = "Unknown"

                stats[model]["processed"] += 1
                pred = (r.get("predicted") or "").strip()
                gt = (r.get("ground_truth") or "").strip()

                if pred and pred.lower() != "unknown":
                    stats[model]["recognized"] += 1
                    if gt:
                        stats[model]["with_gt"] += 1
                        if norm(pred) == norm(gt):
                            stats[model]["tp"] += 1
                        else:
                            stats[model]["fp"] += 1
                    else:
                        stats[model]["fp"] += 1
                else:
                    if gt:
                        stats[model]["with_gt"] += 1
    return stats

def build_dataframe(stats):
    rows = []
    for m, s in stats.items():
        processed = s["processed"]
        recognized = s["recognized"]
        tp = s["tp"]
        with_gt = s["with_gt"]
        precision = (tp / recognized) if recognized else float("nan")
        recall = (tp / with_gt) if with_gt else float("nan")
        f1 = (
            (2 * precision * recall / (precision + recall))
            if (precision and recall and not math.isnan(precision) and not math.isnan(recall))
            else float("nan")
        )
        recognition_rate = recognized / processed if processed else 0.0
        rows.append({
            "Model": m,
            "Processed": processed,
            "Recognized": recognized,
            "Recognition Rate": recognition_rate,
            "True Positives": tp,
            "Precision": precision if not math.isnan(precision) else None,
            "Recall": recall if not math.isnan(recall) else None,
            "F1 Score": f1 if not math.isnan(f1) else None
        })

    df = pd.DataFrame(rows).sort_values("Model")
    # ✅ Remove any 'Unknown' rows automatically
    df = df[df["Model"].str.lower() != "unknown"]
    return df

def plot_metrics(df: pd.DataFrame, out_png: Path):
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    # Left: Precision / Recall / F1 grouped bar
    ax = axes[0]
    metrics_df = df.set_index("Model")[["Precision", "Recall", "F1 Score"]].fillna(0.0)
    metrics_df.plot(kind="bar", ax=ax, rot=0, color=["#2b8cbe", "#7b3294", "#fdae61"])
    ax.set_ylim(0, 1)
    ax.set_ylabel("Score (0–1)")
    ax.set_title("Precision / Recall / F1 by Model")
    for p in ax.patches:
        h = p.get_height()
        if h > 0:
            ax.annotate(f"{h:.2f}", (p.get_x() + p.get_width() / 2., h), ha="center", va="bottom", fontsize=8)

    # Right: Processed and Recognition Rate
    ax2 = axes[1]
    bar = ax2.bar(df["Model"], df["Processed"], color="#d9d9d9")
    ax2.set_ylabel("Processed (count)", color="#444")
    ax2.set_title("Processed Count & Recognition Rate")
    ax2.tick_params(axis="y", labelcolor="#444")
    for i, v in enumerate(df["Processed"]):
        ax2.text(i, v + max(1, int(v * 0.01)), str(v), ha="center", fontsize=8)

    ax3 = ax2.twinx()
    ax3.plot(df["Model"], df["Recognition Rate"], color="#e6550d", marker="o", linewidth=2)
    ax3.set_ylabel("Recognition Rate (fraction)", color="#e6550d")
    ax3.set_ylim(0, 1)
    for i, v in enumerate(df["Recognition Rate"]):
        ax3.text(i, v + 0.02, f"{v:.2f}", ha="center", color="#e6550d", fontsize=8)

    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    try:
        plt.show()
    except Exception:
        pass

def main():
    files = latest_csvs()
    if not files:
        print("No CSV logs found in", LOGS)
        return

    stats = collect_stats(files)
    df = build_dataframe(stats)

    # Display formatted table like your reference
    pd.set_option("display.width", 160)
    pd.set_option("display.precision", 4)

    print("\n" + "="*80)
    print("TABLE IV. PERFORMANCE OF BASE MODELS".center(80))
    print("="*80)
    print(df[["Model", "Precision", "Recall", "F1 Score", "Recognition Rate"]].to_string(index=False))
    print("="*80)

    out_png = LOGS / f"prf_summary_{pd.Timestamp.now().strftime('%Y%m%d_%H%M%S')}.png"
    plot_metrics(df, out_png)
    print("Saved plot to", out_png)

if __name__ == "__main__":
    main()
