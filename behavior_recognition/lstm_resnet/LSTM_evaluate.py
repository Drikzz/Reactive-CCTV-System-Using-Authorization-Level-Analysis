"""Evaluation script for LSTM behavior recognition model.

Features:
- Loads saved checkpoint (best_model.pth or last_model.pth)
- Reconstructs model from saved config & classes
- Evaluates accuracy on the test split
- Optional comparison with a freshly initialized (untrained) model
- Auto-detects model directory when --model-dir is omitted (if exactly one present)

Usage examples (PowerShell):
    # Auto-detect single model directory under models/
    python -m behavior_recognition.LSTM_evaluate --checkpoint best

    # Explicit directory
    python -m behavior_recognition.LSTM_evaluate --model-dir models/LSTM_behavior_recognition --checkpoint last --compare-fresh

Expected model directory contents (created by training script):
    models/<model_name>/
        best_model.pth
        last_model.pth
        config.json
        classes.json
"""
from __future__ import annotations
import os
import json
import argparse
from typing import Optional, Tuple, List

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.transforms as T

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


def build_dataloaders_from_cfg(cfg_dict: dict, classes: List[str]):
    # Recreate minimal config object
    cfg = LSTMConfig()
    for k, v in cfg_dict.items():
        if hasattr(cfg, k):
            setattr(cfg, k, v)
    video_label_pairs, _classes = discover_videos(cfg.root_dir, cfg.max_videos_per_class)
    # ensure classes order matches saved (names must match)
    train_pairs, test_pairs = make_splits(video_label_pairs, classes, cfg.train_split, cfg.random_seed)
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


def evaluate(model: nn.Module, loader: DataLoader, device: torch.device) -> float:
    correct = 0
    total = 0
    with torch.no_grad():
        for clips, labels in loader:
            clips = clips.to(device)
            labels = labels.to(device)
            outputs = model(clips)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
    return 100.0 * correct / max(1, total)


def parse_args():
    p = argparse.ArgumentParser(description='Evaluate LSTM behavior recognition model.')
    p.add_argument('--model-dir', default=None, help='Directory containing checkpoint & metadata (best_model.pth, config.json, classes.json). If omitted, auto-detect if exactly one models/* has config.json.')
    p.add_argument('--checkpoint', choices=['best','last','file'], default='best', help='Which checkpoint to load (best / last / file)')
    p.add_argument('--checkpoint-path', default=None, help='Explicit checkpoint path if --checkpoint file')
    p.add_argument('--compare-fresh', action='store_true', help='Also evaluate an uninitialized model for baseline comparison')
    p.add_argument('--device', default='cuda' if torch.cuda.is_available() else 'cpu')
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
    args = parse_args()
    device = torch.device(args.device)
    model_dir = auto_discover_model_dir(args.model_dir)
    cfg_dict, classes = load_metadata(model_dir)
    test_loader, cfg = build_dataloaders_from_cfg(cfg_dict, classes)

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
    loaded_acc = evaluate(loaded_model, test_loader, device)
    print(f"Loaded model accuracy: {loaded_acc:.2f}%")

    if args.compare_fresh:
        print('Evaluating fresh (untrained) model for baseline...')
        baseline_model = fresh_model(cfg_dict, len(classes), device)
        baseline_acc = evaluate(baseline_model, test_loader, device)
        print(f"Fresh model accuracy: {baseline_acc:.2f}%")

    # Optional comparison to last vs best
    if args.checkpoint == 'best':
        last_path = os.path.join(model_dir, 'last_model.pth')
        if os.path.isfile(last_path):
            try:
                last_model_loaded = load_model(cfg_dict, len(classes), last_path, device)
                last_acc = evaluate(last_model_loaded, test_loader, device)
                print(f"Last model accuracy: {last_acc:.2f}% (difference vs best: {loaded_acc - last_acc:+.2f} pp)")
            except Exception as e:
                print(f"Warning: could not evaluate last model: {e}")

    print('Evaluation complete.')


if __name__ == '__main__':
    main()
