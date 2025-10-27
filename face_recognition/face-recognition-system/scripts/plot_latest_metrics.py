"""
Load the latest evaluation CSVs from the logs folder and plot comparison graphs for up to 3 models.
Saves a PNG to the same logs folder and attempts to show the plots.

Usage (from repo root, venv active):
    python .\scripts\plot_latest_metrics.py --logs "face_recognition\face-recognition-system\data\logs" --n 12

"""
from pathlib import Path
import argparse
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from datetime import datetime

def _norm(s):
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

def guess_model_from_filename(fn):
    name = fn.name.lower()
    if "facenet" in name:
        return "FaceNet"
    if "arcface" in name:
        return "ArcFace"
    if "dlib" in name or "dlib" in name:
        return "Dlib"
    if "model_comparison" in name:
        # common file containing multiple models; will rely on 'model' column
        return None
    return None

def read_csv_try(path):
    try:
        return pd.read_csv(path, encoding="utf-8")
    except Exception:
        try:
            return pd.read_csv(path, encoding="latin1")
        except Exception:
            return None

def find_time_col(df):
    for c in ("inference_time_ms", "time", "time_ms", "inference_time"):
        if c in df.columns:
            return c
    # fallback: any numeric column named like time
    for c in df.columns:
        if "time" in c.lower():
            return c
    return None

def find_conf_col(df):
    for c in ("confidence", "conf", "score"):
        if c in df.columns:
            return c
    for c in df.columns:
        if "conf" in c.lower() or "score" in c.lower():
            return c
    return None

def find_pred_col(df):
    for c in ("predicted","pred","name","prediction"):
        if c in df.columns:
            return c
    return None

def find_gt_col(df):
    for c in ("ground_truth","gt","groundtruth","label","name_gt"):
        if c in df.columns:
            return c
    return None

def collect_metrics_from_file(path):
    df = read_csv_try(path)
    if df is None:
        return []
    timestamp = datetime.fromtimestamp(path.stat().st_mtime)
    model_hint = guess_model_from_filename(path)
    rows = []
    if "model" in df.columns:
        models = df["model"].fillna("unknown").unique()
    elif model_hint:
        models = [model_hint]
    else:
        # try to infer from a predicted column values
        models = ["unknown"]
    time_col = find_time_col(df)
    conf_col = find_conf_col(df)
    pred_col = find_pred_col(df)
    gt_col = find_gt_col(df)
    for m in models:
        if "model" in df.columns:
            sub = df[df["model"] == m].copy()
        else:
            sub = df.copy()
        processed = len(sub)
        if processed == 0:
            continue
        # predicted handling
        pred_values = sub[pred_col].astype(str) if pred_col and pred_col in sub.columns else sub.get("predicted", pd.Series([""]*len(sub))).astype(str)
        recognized_mask = pred_values.str.strip().str.lower() != "unknown"
        recognized = recognized_mask.sum()
        recognition_rate = recognized / processed if processed else np.nan
        # ground truth accuracy
        if gt_col and gt_col in sub.columns:
            gt_values = sub[gt_col].astype(str)
            # consider empty strings as no gt
            has_gt_mask = gt_values.str.strip() != ""
            with_gt = has_gt_mask.sum()
            if with_gt:
                correct_mask = has_gt_mask & (pred_values.str.strip().apply(_norm) == gt_values.str.strip().apply(_norm))
                correct = correct_mask.sum()
                accuracy_gt = correct / with_gt if with_gt else np.nan
            else:
                correct = 0
                accuracy_gt = np.nan
        else:
            with_gt = 0
            correct = 0
            accuracy_gt = np.nan
        # time/conf
        avg_time_ms = float(sub[time_col].dropna().astype(float).mean())*1000.0 if time_col and pd.api.types.is_float_dtype(sub[time_col].dropna()) and ("time" in time_col and "ms" not in time_col) else (float(sub[time_col].dropna().astype(float).mean()) if time_col and sub[time_col].dropna().size>0 else np.nan)
        # above tries to keep ms when already ms; best-effort
        try:
            if time_col and "ms" in time_col:
                avg_time_ms = float(sub[time_col].dropna().astype(float).mean())
        except Exception:
            pass
        if time_col is None:
            avg_time_ms = np.nan
        avg_conf = float(sub[conf_col].dropna().astype(float).mean()) if conf_col and sub[conf_col].dropna().size>0 else np.nan
        rows.append({
            "file": path.name,
            "path": str(path),
            "timestamp": timestamp,
            "model": m if m else "unknown",
            "processed": processed,
            "recognized": int(recognized),
            "recognition_rate": float(recognition_rate),
            "with_gt": int(with_gt),
            "correct": int(correct),
            "accuracy_gt": float(accuracy_gt) if not np.isnan(accuracy_gt) else None,
            "avg_time_ms": float(avg_time_ms) if not pd.isna(avg_time_ms) else None,
            "avg_confidence": float(avg_conf) if not pd.isna(avg_conf) else None
        })
    return rows

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=r"face_recognition\face-recognition-system\data\logs")
    p.add_argument("--n", type=int, default=12, help="number of latest log files to inspect")
    p.add_argument("--out", default="", help="optional output PNG path")
    args = p.parse_args()

    logs_dir = Path(args.logs)
    if not logs_dir.exists():
        print("Logs folder not found:", logs_dir); return

    # collect candidate CSVs
    csvs = sorted([p for p in logs_dir.glob("*.csv")], key=lambda x: x.stat().st_mtime, reverse=True)[:args.n]
    if not csvs:
        print("No CSV logs found in", logs_dir); return

    all_rows = []
    for f in csvs:
        try:
            rows = collect_metrics_from_file(f)
            all_rows.extend(rows)
        except Exception as e:
            print("Failed to parse", f, e)

    if not all_rows:
        print("No metric rows extracted"); return

    metrics_df = pd.DataFrame(all_rows)
    # convert timestamp to sortable
    metrics_df = metrics_df.sort_values("timestamp")

    # Debug: show what was read
    print("[DEBUG] metrics_df.head():")
    print(metrics_df.head(10).to_string(index=False))
    print("[DEBUG] dtypes:")
    print(metrics_df.dtypes)
    print("[DEBUG] null counts:")
    print(metrics_df.isna().sum())

    # replace NaN with 0 for plotting (so lines/bars appear)
    metrics_df_plot = metrics_df.copy()
    metrics_df_plot[["accuracy_gt","recognition_rate","avg_confidence","avg_time_ms"]] = \
        metrics_df_plot[["accuracy_gt","recognition_rate","avg_confidence","avg_time_ms"]].fillna(0.0)

    # ensure timestamp is datetime and sorted
    metrics_df_plot["timestamp"] = pd.to_datetime(metrics_df_plot["timestamp"])
    metrics_df_plot = metrics_df_plot.sort_values("timestamp")

    # limit to up to 3 models present (user requested compare 3 models)
    models = list(metrics_df_plot["model"].unique())
    if len(models) > 3:
        # keep top 3 by overall processed count
        totals = metrics_df.groupby("model")["processed"].sum().sort_values(ascending=False)
        keep = list(totals.index[:3])
        metrics_df_plot = metrics_df_plot[metrics_df_plot["model"].isin(keep)]
        models = keep

    # plotting (use pivot so each model becomes a series)
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2,2, figsize=(14,9))
    axes = axes.flatten()

    def plot_line_metric(ax, metric, ylabel):
        pivot = metrics_df_plot.pivot_table(index="timestamp", columns="model", values=metric, aggfunc="mean")
        if pivot.empty:
            ax.text(0.5,0.5,"no data",ha="center")
            return
        pivot.plot(ax=ax, marker="o", linewidth=2)
        ax.set_title(ylabel)
        ax.set_ylabel(ylabel)
        ax.set_xlabel("Run timestamp")
        ax.tick_params(axis="x", rotation=25)
        ax.legend(title="model", loc="best")

    # accuracy (line)
    plot_line_metric(axes[0], "accuracy_gt", "GT Accuracy")

    # recognition rate (grouped bar per run) - easier to read when multiple runs
    ax = axes[1]
    pivot_rr = metrics_df_plot.pivot_table(index="timestamp", columns="model", values="recognition_rate", aggfunc="mean").fillna(0.0)
    if pivot_rr.empty:
        ax.text(0.5,0.5,"no data",ha="center")
    else:
        pivot_rr.plot(kind="bar", ax=ax)
        ax.set_title("Recognition rate per run (bar)")
        ax.set_ylabel("Recognition rate")
        ax.set_xlabel("Run timestamp")
        ax.tick_params(axis="x", rotation=25)

    # avg confidence (line)
    plot_line_metric(axes[2], "avg_confidence", "Average confidence")

    # avg time (line)
    plot_line_metric(axes[3], "avg_time_ms", "Average inference time (ms)")

    plt.tight_layout()

    out_png = Path(args.out) if args.out else (logs_dir / f"metrics_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    fig.savefig(str(out_png), dpi=200)
    print("Saved plot to", out_png)

    try:
        plt.show()
    except Exception:
        pass

if __name__ == "__main__":
    main()