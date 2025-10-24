import argparse
from pathlib import Path
from datetime import datetime
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns

LOGS_DEFAULT = Path("face_recognition/face-recognition-system/data/logs")

def norm(s):
    return "".join(ch.lower() for ch in (s or "") if ch.isalnum())

def guess_model_from_filename(p: Path):
    n = p.name.lower()
    if "facenet" in n: return "FaceNet"
    if "arcface" in n: return "ArcFace"
    if "dlib" in n or "dlib" in n.name: return "Dlib"
    if "model_comparison" in n: return None
    return None

def find_time_col(df):
    for c in ("inference_time_ms","inference_time","time_ms","time"):
        if c in df.columns: return c
    for c in df.columns:
        if "time" in c.lower(): return c
    return None

def find_conf_col(df):
    for c in ("confidence","conf","score"):
        if c in df.columns: return c
    for c in df.columns:
        if "conf" in c.lower() or "score" in c.lower(): return c
    return None

def find_pred_col(df):
    for c in ("predicted","pred","name"):
        if c in df.columns: return c
    return None

def find_gt_col(df):
    for c in ("ground_truth","gt","label","name"):
        if c in df.columns: return c
    return None

def read_csv_try(path: Path):
    for enc in ("utf-8","latin1"):
        try:
            return pd.read_csv(path, encoding=enc)
        except Exception:
            pass
    return None

def collect_metrics(logs_dir: Path):
    rows = []
    csvs = sorted(list(logs_dir.glob("*.csv")), key=lambda p: p.stat().st_mtime, reverse=True)
    for p in csvs:
        df = read_csv_try(p)
        if df is None:
            continue
        ts = datetime.fromtimestamp(p.stat().st_mtime)
        model_hint = guess_model_from_filename(p)
        models_in_file = []
        if "model" in df.columns:
            models_in_file = list(pd.Series(df["model"].fillna("")).unique())
        elif model_hint:
            models_in_file = [model_hint]
        else:
            models_in_file = ["unknown"]
        time_col = find_time_col(df)
        conf_col = find_conf_col(df)
        pred_col = find_pred_col(df)
        gt_col = find_gt_col(df)
        for m in models_in_file:
            sub = df[df["model"]==m].copy() if "model" in df.columns else df.copy()
            processed = len(sub)
            if processed == 0:
                continue
            # predicted handling
            if pred_col and pred_col in sub.columns:
                pred_values = sub[pred_col].astype(str).fillna("")
            else:
                pred_values = sub.get("predicted", pd.Series([""]*len(sub))).astype(str).fillna("")
            recognized_mask = pred_values.str.strip().str.lower() != "unknown"
            recognized = int(recognized_mask.sum())
            recognition_rate = recognized / processed if processed else np.nan
            # ground truth accuracy
            with_gt = 0; correct = 0; accuracy_gt = np.nan
            if gt_col and gt_col in sub.columns:
                gt_vals = sub[gt_col].astype(str).fillna("")
                has_gt_mask = gt_vals.str.strip() != ""
                with_gt = int(has_gt_mask.sum())
                if with_gt:
                    preds = pred_values
                    correct = int(((preds.str.strip().apply(norm) == gt_vals.str.strip().apply(norm)) & has_gt_mask).sum())
                    accuracy_gt = correct / with_gt if with_gt else np.nan
            # avg_time_ms
            avg_time_ms = None
            if time_col and time_col in sub.columns:
                try:
                    times = pd.to_numeric(sub[time_col], errors="coerce").dropna()
                    if not times.empty:
                        # if column name suggests seconds -> convert
                        if "ms" in time_col.lower():
                            avg_time_ms = float(times.mean())
                        else:
                            avg_time_ms = float(times.mean()) * 1000.0
                except Exception:
                    avg_time_ms = None
            # avg confidence for recognized
            avg_conf = None
            if conf_col and conf_col in sub.columns:
                try:
                    confs = pd.to_numeric(sub.loc[recognized_mask, conf_col], errors="coerce").dropna()
                    if not confs.empty:
                        avg_conf = float(confs.mean())
                except Exception:
                    avg_conf = None
            rows.append({
                "file": p.name,
                "path": str(p),
                "timestamp": ts,
                "model": m if m else (model_hint or "unknown"),
                "processed": processed,
                "recognized": recognized,
                "recognition_rate": recognition_rate,
                "with_gt": with_gt,
                "correct": correct,
                "accuracy_gt": accuracy_gt,
                "avg_time_ms": avg_time_ms,
                "avg_confidence": avg_conf
            })
    return pd.DataFrame(rows)

def plot_metrics(df: pd.DataFrame, out_png: Path):
    if df.empty:
        print("No metric rows found to plot"); return
    df = df.sort_values("timestamp")
    # prefer only common models: FaceNet, ArcFace, Dlib if present
    models_priority = ["FaceNet","ArcFace","Dlib"]
    present = [m for m in models_priority if m in df["model"].unique()]
    if not present:
        present = list(df["model"].unique())[:3]
    df = df[df["model"].isin(present)].copy()
    # fill missing numerics for plotting (but keep accuracy_gt NaN to show gaps)
    df_plot = df.copy()
    df_plot[["recognition_rate","avg_confidence","avg_time_ms"]] = df_plot[["recognition_rate","avg_confidence","avg_time_ms"]].fillna(0.0)
    sns.set(style="whitegrid")
    fig, axes = plt.subplots(2,2, figsize=(16,10))
    axes = axes.flatten()
    palette = sns.color_palette("tab10", n_colors=len(present))
    # accuracy line
    ax = axes[0]
    for i,m in enumerate(present):
        subset = df_plot[df_plot["model"]==m]
        ax.plot(subset["timestamp"], subset["accuracy_gt"].fillna(np.nan), marker="o", linewidth=3, markersize=8, label=m, color=palette[i])
    ax.set_title("GT Accuracy over time")
    ax.set_ylabel("Accuracy (fraction)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    # recognition rate bar grouped
    ax = axes[1]
    pivot_rr = df_plot.pivot_table(index="timestamp", columns="model", values="recognition_rate", aggfunc="mean").fillna(0.0)
    if not pivot_rr.empty:
        pivot_rr.plot(kind="bar", ax=ax, color=palette)
    ax.set_title("Recognition rate per run (bar)")
    ax.set_ylabel("Recognition rate")
    ax.tick_params(axis="x", rotation=30)
    # avg confidence line
    ax = axes[2]
    for i,m in enumerate(present):
        subset = df_plot[df_plot["model"]==m]
        ax.plot(subset["timestamp"], subset["avg_confidence"], marker="o", linewidth=3, markersize=8, label=m, color=palette[i])
    ax.set_title("Average confidence (recognized)")
    ax.set_ylabel("Avg confidence")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    # avg time line
    ax = axes[3]
    for i,m in enumerate(present):
        subset = df_plot[df_plot["model"]==m]
        ax.plot(subset["timestamp"], subset["avg_time_ms"], marker="o", linewidth=3, markersize=8, label=m, color=palette[i])
    ax.set_title("Average inference time (ms)")
    ax.set_ylabel("Avg time (ms)")
    ax.legend()
    ax.tick_params(axis="x", rotation=30)
    plt.tight_layout()
    fig.savefig(out_png, dpi=200)
    print("Saved plot to", out_png)
    try:
        plt.show()
    except Exception:
        pass

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--logs", default=str(LOGS_DEFAULT))
    p.add_argument("--out", default="")
    args = p.parse_args()
    logs = Path(args.logs)
    if not logs.exists():
        print("Logs folder not found:", logs); return
    df = collect_metrics(logs)
    out_png = Path(args.out) if args.out else (logs / f"all_models_summary_{datetime.now().strftime('%Y%m%d_%H%M%S')}.png")
    plot_metrics(df, out_png)

if __name__ == "__main__":
    main()