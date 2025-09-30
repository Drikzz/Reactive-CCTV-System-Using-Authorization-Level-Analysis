"""Enhanced evaluation script for LSTM behavior recognition model.

Adds richer metrics while preserving original behavior:
    - Load checkpoint (best / last / explicit file)
    - Reconstruct model from saved config + classes
    - Evaluate on the same test split logic as training (re-discovered videos)
    - Optional: compare against fresh (untrained) model
    - Optional: verify equivalence of model.forward vs an "inference path" (manual CNN -> LSTM -> classifier) used in streaming inference
    - Produce per-class accuracy & confusion matrix (text) and top-k accuracies
    - Optional CSV of per-clip predictions
    - Auto-detect model directory if exactly one under models/
    - Allow overriding dataset root / cache at evaluation time without editing training script

Usage examples (PowerShell):
    # Basic (auto-detect model dir)
    python -m behavior_recognition.lstm_resnet.LSTM_evaluate --checkpoint best

    # With extra metrics & CSV
    python -m behavior_recognition.lstm_resnet.LSTM_evaluate --model-dir models/LSTM_behavior_recognition \
            --checkpoint last --pred-csv eval_preds.csv --topk 1 3 5 --confusion

    # Verify inference path equivalence
    python -m behavior_recognition.lstm_resnet.LSTM_evaluate --verify-inference-path

Output CSV columns:
    clip_path,label_true,label_pred,correct,topk_hit,prob_pred
"""
from __future__ import annotations
import os
import json
import argparse
import sys
from typing import Optional, Tuple, List, Sequence

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T
import numpy as np
import time
import csv
from datetime import datetime

# Optional plotting libs (lazy import later to keep base run lightweight)
try:
    import matplotlib.pyplot as plt  # type: ignore
except Exception:  # pragma: no cover
    plt = None  # type: ignore
try:
    import seaborn as sns  # type: ignore
except Exception:  # pragma: no cover
    sns = None  # type: ignore

# Import dataset & model definitions from training file (reuse code if available)
try:
    from .LSTM_behavior_recognition import (
        LSTMConfig,
        BehaviorVideoDataset,
        CNNLSTM,
        discover_videos,
        make_splits,
    )
except ImportError:
    # Fallback if run as standalone script (module relative import may fail)
    from LSTM_behavior_recognition import (  # type: ignore
        LSTMConfig,
        BehaviorVideoDataset,
        CNNLSTM,
        discover_videos,
        make_splits,
    )


def load_metadata(model_dir: str):
    config_path = os.path.join(model_dir, 'config.json')
    classes_path = os.path.join(model_dir, 'classes.json')
    if not os.path.isfile(config_path):
        raise FileNotFoundError(f"Missing config.json in {model_dir}")
    if not os.path.isfile(classes_path):
        raise FileNotFoundError(f"Missing classes.json in {model_dir}")
    with open(config_path, 'r', encoding='utf-8') as f:
        cfg_dict = json.load(f)
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes_map = json.load(f)
    # classes.json stored as {index: class_name}
    # Ensure ordering by index
    classes = [classes_map[str(i)] if str(i) in classes_map else classes_map[i] for i in sorted(map(int, classes_map.keys()))]
    return cfg_dict, classes


def build_dataloaders_from_cfg(cfg_dict: dict, classes: List[str], override_root: Optional[str] = None, override_cache: Optional[str] = None):
    # Recreate minimal config object
    cfg = LSTMConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    if override_root:
        cfg.root_dir = override_root
    if override_cache:
        cfg.cache_root = override_cache
    video_label_pairs, discovered_classes = discover_videos(cfg.root_dir, cfg.max_videos_per_class)
    # Remap discovered labels to saved classes list; skip any unknown classes gracefully
    saved_index = {name: i for i, name in enumerate(classes)}
    remapped: List[Tuple[str,int]] = []
    skipped_unknown = 0
    for path, lab in video_label_pairs:
        if 0 <= lab < len(discovered_classes):
            cname = discovered_classes[lab]
            if cname in saved_index:
                remapped.append((path, saved_index[cname]))
            else:
                skipped_unknown += 1
        else:
            skipped_unknown += 1
    if skipped_unknown > 0:
        print(f"[WARN] Skipped {skipped_unknown} clip(s) whose class is not in saved model classes.json.")
    if not remapped:
        raise RuntimeError("After remapping, no videos matched the saved classes. Ensure dataset structure aligns with classes.json.")
    train_pairs, test_pairs = make_splits(remapped, classes, cfg.train_split, cfg.random_seed)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((cfg.img_size, cfg.img_size)),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    test_ds = BehaviorVideoDataset(
        test_pairs, classes, cfg.sequence_length, transform, cfg.img_size,
        cfg.frame_sampling, show_progress=False,
        cache_root=cfg.cache_root, enable_cache=cfg.enable_cache, dataset_root=cfg.root_dir
    )
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return test_loader, cfg


def load_model(cfg_dict: dict, num_classes: int, checkpoint_path: str, device: torch.device) -> CNNLSTM:
    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=cfg_dict.get('hidden_size', 256),
        num_layers=cfg_dict.get('num_layers', 2),
        bidirectional=cfg_dict.get('bidirectional', False),
        dropout=cfg_dict.get('dropout', 0.3),
        freeze_cnn=cfg_dict.get('freeze_cnn', True)
    )
    ckpt = torch.load(checkpoint_path, map_location='cpu')
    state = ckpt.get('model_state', ckpt)  # support plain state_dict
    model.load_state_dict(state, strict=True)
    model.to(device)
    model.eval()
    return model


def fresh_model(cfg_dict: dict, num_classes: int, device: torch.device) -> CNNLSTM:
    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=cfg_dict.get('hidden_size', 256),
        num_layers=cfg_dict.get('num_layers', 2),
        bidirectional=cfg_dict.get('bidirectional', False),
        dropout=cfg_dict.get('dropout', 0.3),
        freeze_cnn=cfg_dict.get('freeze_cnn', True)
    ).to(device)
    model.eval()
    return model


def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
    classes: Optional[List[str]] = None,
    topk: Sequence[int] = (1,),
    collect_preds: bool = False,
    verify_inference_path: bool = False,
) -> dict:
    """Run evaluation and return metrics.

    Returns dict with keys: accuracy, topk, per_class, confusion, predictions (optional)
    """
    topk = sorted(set(int(k) for k in topk if k > 0))
    n_classes = None if classes is None else len(classes)
    correct = 0
    total = 0
    # per-class counts
    per_class_correct = None if classes is None else [0]*len(classes)
    per_class_total = None if classes is None else [0]*len(classes)
    # confusion matrix
    confusion = None if classes is None else np.zeros((len(classes), len(classes)), dtype=np.int64)
    # predictions list
    pred_rows = [] if collect_preds else None

    # top-k hits aggregate
    topk_hits = {k: 0 for k in topk}

    t_start = time.time()
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)  # (B,C)

            if verify_inference_path:
                # Recreate inference-style path: CNN on each frame then LSTM + classifier
                # Expect model is CNNLSTM
                try:
                    B, T, C, H, W = clips.shape
                    flat = clips.view(B*T, C, H, W)
                    feats = model.cnn(flat)  # (B*T,512,1,1)
                    feats = feats.view(B, T, -1)
                    lstm_out, _ = model.lstm(feats)
                    last = lstm_out[:, -1, :]
                    outputs_inf = model.classifier(last)
                    if not torch.allclose(outputs, outputs_inf, atol=1e-5, rtol=1e-4):
                        print("Warning: forward() logits differ from reconstructed inference path (max abs diff {:.4e})".format(
                            (outputs - outputs_inf).abs().max().item()
                        ))
                except Exception as e:
                    print(f"Inference-path verification failed: {e}")

            probs = torch.softmax(outputs, dim=1)
            confs, preds = probs.max(1)
            total += labels.size(0)
            correct += preds.eq(labels).sum().item()

            # top-k
            if topk:
                sorted_probs, sorted_idx = probs.topk(k=max(topk), dim=1)
                for k in topk:
                    # match if true label in first k indices
                    hits = (sorted_idx[:, :k] == labels.unsqueeze(1)).any(dim=1).sum().item()
                    topk_hits[k] += hits

            if per_class_total is not None:
                for lbl, pred in zip(labels.tolist(), preds.tolist()):
                    per_class_total[lbl] += 1
                    if lbl == pred:
                        per_class_correct[lbl] += 1
                    if confusion is not None:
                        confusion[lbl, pred] += 1

            if pred_rows is not None:
                for i in range(labels.size(0)):
                    row = {
                        'clip_index': len(pred_rows),
                        'label_true': int(labels[i].item()),
                        'label_pred': int(preds[i].item()),
                        'correct': int(preds[i].item() == labels[i].item()),
                        'prob_pred': float(confs[i].item()),
                    }
                    pred_rows.append(row)

    acc = 100.0 * correct / max(1, total)
    topk_acc = {k: 100.0 * topk_hits[k] / max(1, total) for k in topk}
    per_class_stats = None
    prf_aggregates = None
    if per_class_total is not None and confusion is not None:
        per_class_stats = []
        cm = confusion
        # Pre-compute sums
        row_sums = cm.sum(axis=1)  # support (actual)
        col_sums = cm.sum(axis=0)  # predicted counts
        diag = np.diag(cm)
        macro_p_list = []
        macro_r_list = []
        macro_f1_list = []
        supports = []
        for i, tot in enumerate(row_sums):
            tp = diag[i]
            fp = col_sums[i] - tp
            fn = tot - tp
            precision = float(tp / (tp + fp)) if (tp + fp) > 0 else 0.0
            recall = float(tp / (tp + fn)) if (tp + fn) > 0 else 0.0
            f1 = float(2*precision*recall / (precision + recall)) if (precision + recall) > 0 else 0.0
            acc_cls = float(tp / tot) * 100.0 if tot > 0 else 0.0
            per_class_stats.append({
                'index': i,
                'class': classes[i] if classes and i < len(classes) else str(i),
                'total': int(tot),
                'correct': int(tp),
                'accuracy': acc_cls,
                'precision': precision,
                'recall': recall,
                'f1': f1,
            })
            if tot > 0:
                macro_p_list.append(precision)
                macro_r_list.append(recall)
                macro_f1_list.append(f1)
                supports.append(tot)
        supports_arr = np.array(supports, dtype=np.float64) if supports else np.array([1.0])
        weight_norm = supports_arr.sum() if supports else 1.0
        # Macro averages (exclude zero-support classes)
        macro_precision = float(np.mean(macro_p_list)) if macro_p_list else 0.0
        macro_recall = float(np.mean(macro_r_list)) if macro_r_list else 0.0
        macro_f1 = float(np.mean(macro_f1_list)) if macro_f1_list else 0.0
        # Weighted averages
        weighted_precision = float(np.average(macro_p_list, weights=supports_arr)) if macro_p_list else 0.0
        weighted_recall = float(np.average(macro_r_list, weights=supports_arr)) if macro_r_list else 0.0
        weighted_f1 = float(np.average(macro_f1_list, weights=supports_arr)) if macro_f1_list else 0.0
        # Micro (aggregate)
        tp_total = float(diag.sum())
        micro_precision = tp_total / max(1.0, col_sums.sum())
        micro_recall = tp_total / max(1.0, row_sums.sum())
        micro_f1 = 2*micro_precision*micro_recall / max(1e-12, (micro_precision + micro_recall))
        prf_aggregates = {
            'macro': {'precision': macro_precision, 'recall': macro_recall, 'f1': macro_f1},
            'weighted': {'precision': weighted_precision, 'recall': weighted_recall, 'f1': weighted_f1},
            'micro': {'precision': micro_precision, 'recall': micro_recall, 'f1': micro_f1},
        }

    return {
        'accuracy': acc,
        'samples': total,
        'topk': topk_acc,
        'per_class': per_class_stats,
        'confusion': confusion,
        'predictions': pred_rows,
        'elapsed_sec': time.time() - t_start,
        'prf': prf_aggregates,
    }


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate LSTM behavior recognition model (enhanced).')
    p.add_argument('--model-dir', default=None, help='Directory containing checkpoint & metadata (best_model.pth, config.json, classes.json). If omitted, auto-detect.')
    p.add_argument('--checkpoint', choices=['best','last','file'], default='best', help="Checkpoint to load (default: best). Omit this flag to automatically use best_model.pth.")
    p.add_argument('--checkpoint-path', default=None, help='Explicit checkpoint path if --checkpoint file')
    p.add_argument('--compare-fresh', action='store_true', help='Also evaluate an uninitialized model for baseline comparison')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu', help='Device to use (default auto)')
    p.add_argument('--topk', nargs='*', type=int, default=[1], help='Top-k accuracies to compute (e.g. --topk 1 3 5)')
    p.add_argument('--pred-csv', default=None, help='If set, write per-clip predictions to this CSV file')
    p.add_argument('--confusion', action='store_true', help='Print confusion matrix')
    p.add_argument('--per-class', action='store_true', help='Print per-class accuracy table')
    p.add_argument('--verify-inference-path', action='store_true', help='Check that forward() matches manual inference path logits')
    p.add_argument('--data-root', default=None, help='Override dataset root (cfg.root_dir) for evaluation only')
    p.add_argument('--cache-root', default=None, help='Override cache root (cfg.cache_root) for evaluation only')
    p.add_argument('--cm-fig', default=None, help='Path to save confusion matrix heatmap (PNG). Use with --confusion')
    p.add_argument('--cm-normalize', choices=['none','true','pred','all'], default='true', help='Normalization mode for confusion matrix plot')
    p.add_argument('--pr-bar-fig', default=None, help='Path to save per-class precision/recall/F1 bar chart (PNG); requires --per-class')
    p.add_argument('--fig-dpi', type=int, default=130, help='DPI for saved figures (default: 130)')
    return p.parse_args()


def auto_discover_model_dir(explicit: Optional[str]) -> str:
    """Return a model directory.

    If explicit provided, validate & return.
    Else search ./models/* for exactly one directory containing config.json & classes.json.
    """
    if explicit:
        if not os.path.isdir(explicit):
            raise FileNotFoundError(f"Specified --model-dir not found: {explicit}")
        return explicit
    root = os.getcwd()
    models_root = os.path.join(root, 'models')
    if not os.path.isdir(models_root):
        raise FileNotFoundError("No models directory found and --model-dir not specified.")
    candidates = []
    for name in sorted(os.listdir(models_root)):
        path = os.path.join(models_root, name)
        if os.path.isdir(path) and os.path.isfile(os.path.join(path, 'config.json')):
            # require classes.json too
            if os.path.isfile(os.path.join(path, 'classes.json')):
                candidates.append(path)
    if len(candidates) == 0:
        raise FileNotFoundError("No model directories with config.json & classes.json found under models/. Specify --model-dir.")
    if len(candidates) > 1:
        joined = '\n  '.join(candidates)
        raise ValueError(f"Multiple model directories found; please specify one with --model-dir:\n  {joined}")
    print(f"Auto-detected model directory: {candidates[0]}")
    return candidates[0]


def main():
    # Auto-inject default arguments if user provided none.
    # This allows simply: python LSTM_evaluate.py (or -m ...) to evaluate best model with rich metrics.
    if len(sys.argv) == 1:
        # Replace argv with desired defaults while keeping program name.
        sys.argv = [sys.argv[0], '--checkpoint', 'best', '--per-class', '--confusion']
    args = parse_args()
    device = torch.device(args.device)
    model_dir = auto_discover_model_dir(args.model_dir)
    cfg_dict, classes = load_metadata(model_dir)
    test_loader, cfg = build_dataloaders_from_cfg(cfg_dict, classes, args.data_root, args.cache_root)

    # Determine checkpoint path
    if args.checkpoint == 'best':
        ckpt_path = os.path.join(model_dir, 'best_model.pth')
    elif args.checkpoint == 'last':
        ckpt_path = os.path.join(model_dir, 'last_model.pth')
    else:
        if not args.checkpoint_path:
            raise ValueError('--checkpoint file requires --checkpoint-path')
        ckpt_path = args.checkpoint_path
    if not os.path.isfile(ckpt_path):
        raise FileNotFoundError(f'Checkpoint not found: {ckpt_path}')

    print(f"Loading checkpoint: {ckpt_path}")
    loaded_model = load_model(cfg_dict, len(classes), ckpt_path, device)
    metrics = evaluate(
        loaded_model,
        test_loader,
        device,
        classes=classes,
        topk=args.topk,
        collect_preds=bool(args.pred_csv),
        verify_inference_path=args.verify_inference_path,
    )
    print(f"Loaded model accuracy: {metrics['accuracy']:.2f}% on {metrics['samples']} samples")
    if metrics['topk']:
        topk_str = ', '.join(f"top@{k}={v:.2f}%" for k, v in metrics['topk'].items())
        print(f"Top-k: {topk_str}")
    if args.per_class and metrics['per_class']:
        print("Per-class metrics:")
        for row in metrics['per_class']:
            print(
                f"  [{row['index']:02d}] {row['class']:<30} acc={row['accuracy']:.2f}% "
                f"P={row['precision']:.3f} R={row['recall']:.3f} F1={row['f1']:.3f} "
                f"(correct {row['correct']}/{row['total']})"
            )
    if metrics.get('prf'):
        prf = metrics['prf']
        print("\nAggregate Precision/Recall/F1:")
        for k in ('micro','macro','weighted'):
            if k in prf:
                print(f"  {k:<8} P={prf[k]['precision']:.3f} R={prf[k]['recall']:.3f} F1={prf[k]['f1']:.3f}")
    if args.confusion and metrics['confusion'] is not None:
        print("Confusion matrix (rows=true, cols=pred):")
        cm = metrics['confusion']
        # header (limit width if too large)
        max_print = min(20, cm.shape[0])
        header = '      ' + ' '.join(f"{i:>4d}" for i in range(max_print))
        print(header)
        for i in range(max_print):
            row_cells = ' '.join(f"{int(cm[i,j]):>4d}" for j in range(max_print))
            print(f"{i:>4d}: {row_cells}")
        if cm.shape[0] > max_print:
            print(f"(truncated to first {max_print} classes)")
    if args.pred_csv and metrics['predictions'] is not None:
        os.makedirs(os.path.dirname(args.pred_csv) or '.', exist_ok=True)
        with open(args.pred_csv, 'w', newline='', encoding='utf-8') as f:
            writer = csv.DictWriter(f, fieldnames=list(metrics['predictions'][0].keys()))
            writer.writeheader()
            writer.writerows(metrics['predictions'])
        print(f"Wrote predictions CSV: {args.pred_csv}")

    # ---- Visualization (confusion matrix) ----
    if args.cm_fig and metrics.get('confusion') is not None:
        if plt is None:
            print('Cannot plot confusion matrix: matplotlib not installed.')
        else:
            cm = metrics['confusion'].astype(float)
            norm_mode = args.cm_normalize
            if norm_mode != 'none':
                with np.errstate(divide='ignore', invalid='ignore'):
                    if norm_mode == 'true':
                        sums = cm.sum(axis=1, keepdims=True)
                        cm = np.divide(cm, sums, out=np.zeros_like(cm), where=sums>0)
                    elif norm_mode == 'pred':
                        sums = cm.sum(axis=0, keepdims=True)
                        cm = np.divide(cm, sums, out=np.zeros_like(cm), where=sums>0)
                    elif norm_mode == 'all':
                        total = cm.sum()
                        if total > 0:
                            cm = cm / total
            # Limit for readability
            max_classes_to_plot = 60
            label_subset = classes
            plot_cm = cm
            if cm.shape[0] > max_classes_to_plot:
                plot_cm = cm[:max_classes_to_plot, :max_classes_to_plot]
                label_subset = classes[:max_classes_to_plot] if classes else [str(i) for i in range(max_classes_to_plot)]
                print(f"Confusion matrix truncated to first {max_classes_to_plot} classes for plotting.")
            fig_w = min(20, max(6, 0.35 * plot_cm.shape[0]))
            fig_h = fig_w
            plt.figure(figsize=(fig_w, fig_h))
            if sns is not None:
                sns.heatmap(plot_cm, annot=plot_cm.shape[0] <= 30, fmt='.2f' if norm_mode!='none' else 'g',
                            cmap='viridis', cbar=True, xticks=range(len(label_subset)), yticks=range(len(label_subset)))
            else:
                plt.imshow(plot_cm, cmap='viridis')
                if plot_cm.shape[0] <= 30:
                    for i in range(plot_cm.shape[0]):
                        for j in range(plot_cm.shape[1]):
                            val = plot_cm[i, j]
                            plt.text(j, i, f"{val:.2f}" if norm_mode!='none' else int(val), ha='center', va='center', color='w', fontsize=7)
                plt.colorbar()
            plt.title(f"Confusion Matrix ({norm_mode})")
            plt.xlabel('Predicted')
            plt.ylabel('True')
            if label_subset and len(label_subset) <= 60:
                plt.xticks(ticks=np.arange(len(label_subset))+0.5, labels=[l.split('/')[-1] for l in label_subset], rotation=90, fontsize=8)
                plt.yticks(ticks=np.arange(len(label_subset))+0.5, labels=[l.split('/')[-1] for l in label_subset], rotation=0, fontsize=8)
            plt.tight_layout()
            os.makedirs(os.path.dirname(args.cm_fig) or '.', exist_ok=True)
            plt.savefig(args.cm_fig, dpi=args.fig_dpi)
            plt.close()
            print(f"Saved confusion matrix figure: {args.cm_fig}")

    # ---- Visualization (precision/recall/F1 bar chart) ----
    if args.pr_bar_fig and metrics.get('per_class'):
        if plt is None:
            print('Cannot plot PR/F1 bars: matplotlib not installed.')
        else:
            rows = metrics['per_class']
            names = [r['class'].split('/')[-1] for r in rows]
            precision = [r['precision'] for r in rows]
            recall = [r['recall'] for r in rows]
            f1 = [r['f1'] for r in rows]
            idx = np.arange(len(rows))
            bar_w = 0.25
            fig_w = min(24, max(8, 0.4 * len(rows)))
            plt.figure(figsize=(fig_w, 6))
            plt.bar(idx - bar_w, precision, width=bar_w, label='Precision')
            plt.bar(idx, recall, width=bar_w, label='Recall')
            plt.bar(idx + bar_w, f1, width=bar_w, label='F1')
            plt.xticks(idx, names, rotation=90 if len(rows) > 12 else 45, ha='right', fontsize=8)
            plt.ylim(0, 1.05)
            plt.ylabel('Score')
            plt.title('Per-Class Precision / Recall / F1')
            plt.legend()
            plt.tight_layout()
            os.makedirs(os.path.dirname(args.pr_bar_fig) or '.', exist_ok=True)
            plt.savefig(args.pr_bar_fig, dpi=args.fig_dpi)
            plt.close()
            print(f"Saved PR/F1 bar figure: {args.pr_bar_fig}")

    baseline_acc = None
    if args.compare_fresh:
        print('Evaluating fresh (untrained) model for baseline...')
        baseline_model = fresh_model(cfg_dict, len(classes), device)
        baseline_acc = evaluate(baseline_model, test_loader, device)
        print(f"Fresh model accuracy: {baseline_acc:.2f}%")

    # Optional comparison to last vs best
    last_metrics_acc = None
    if args.checkpoint == 'best':
        last_path = os.path.join(model_dir, 'last_model.pth')
        if os.path.isfile(last_path):
            try:
                last_model_loaded = load_model(cfg_dict, len(classes), last_path, device)
                last_metrics = evaluate(last_model_loaded, test_loader, device, classes=classes, topk=[1])
                print(f"Last model accuracy: {last_metrics['accuracy']:.2f}% (difference vs best: {metrics['accuracy'] - last_metrics['accuracy']:+.2f} pp)")
                last_metrics_acc = last_metrics['accuracy']
            except Exception as e:
                print(f"Warning: could not evaluate last model: {e}")

    # ---- Always persist metrics JSON ----
    try:
        save_dir = model_dir
        os.makedirs(save_dir, exist_ok=True)
        timestamp = datetime.utcnow().strftime('%Y%m%d_%H%M%S')
        metrics_to_save = {
            'timestamp_utc': timestamp,
            'checkpoint_loaded': ckpt_path,
            'args': {
                'checkpoint': args.checkpoint,
                'topk': args.topk,
                'per_class_requested': args.per_class,
                'confusion_requested': args.confusion,
                'verify_inference_path': args.verify_inference_path,
                'data_root_override': args.data_root,
                'cache_root_override': args.cache_root,
            },
            'accuracy': metrics.get('accuracy'),
            'topk': metrics.get('topk'),
            'aggregate_prf': metrics.get('prf'),
            'samples': metrics.get('samples'),
            'elapsed_sec': metrics.get('elapsed_sec'),
            'baseline_accuracy': baseline_acc,
            'last_model_accuracy': last_metrics_acc,
        }
        if args.per_class and metrics.get('per_class') is not None:
            metrics_to_save['per_class'] = metrics['per_class']
        if args.confusion and metrics.get('confusion') is not None:
            # store raw confusion matrix counts (not normalized)
            metrics_to_save['confusion_matrix'] = metrics['confusion'].tolist()
        # Write a timestamped file and update a latest pointer
        ts_file = os.path.join(save_dir, f'eval_metrics_{timestamp}.json')
        with open(ts_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2)
        latest_file = os.path.join(save_dir, 'eval_metrics_latest.json')
        with open(latest_file, 'w', encoding='utf-8') as f:
            json.dump(metrics_to_save, f, indent=2)
        print(f"Saved metrics: {ts_file} (also updated eval_metrics_latest.json)")
    except Exception as e:
        print(f"Warning: failed to save metrics JSON: {e}")

    print('Evaluation complete.')


if __name__ == '__main__':
    main()
