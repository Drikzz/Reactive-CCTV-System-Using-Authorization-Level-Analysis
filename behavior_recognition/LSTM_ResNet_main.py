import os
import sys
import json
import time
import argparse
from collections import deque
from typing import Deque, List, Tuple, Optional

import cv2
import numpy as np
import torch
import torch.nn.functional as F

# Ensure project root is on sys.path so we can import the training module
THIS_DIR = os.path.dirname(os.path.abspath(__file__))
PROJECT_ROOT = os.path.abspath(os.path.join(THIS_DIR, ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

# Import the model class from the training script
try:
    from behavior_recognition.LSTM_behavior_recognition import CNNLSTM
except Exception:
    # Fallback: allow direct import if run from project root without package
    sys.path.insert(0, os.path.abspath(os.path.join(THIS_DIR)))
    from LSTM_behavior_recognition import CNNLSTM  # type: ignore


IMAGENET_MEAN = torch.tensor([0.485, 0.456, 0.406]).view(3, 1, 1)
IMAGENET_STD = torch.tensor([0.229, 0.224, 0.225]).view(3, 1, 1)


def default_model_dir() -> str:
    # Models saved by LSTM_behavior_recognition.py to: models/LSTM_behavior_recognition/*.pth
    # Resolve relative to project root for robustness.
    return os.path.abspath(os.path.join(PROJECT_ROOT, "models", "LSTM_behavior_recognition"))


def find_weights_and_assets(
    explicit_weights: Optional[str] = None,
    explicit_model_dir: Optional[str] = None
) -> Tuple[str, str, str]:
    """
    Returns (weights_path, classes_json_path, config_json_path).
    Prefers best_model.pth, falls back to last_model.pth.
    """
    model_dir = explicit_model_dir or default_model_dir()
    if not os.path.isdir(model_dir):
        raise FileNotFoundError(f"Model directory not found: {model_dir}")

    # Determine weights path
    if explicit_weights:
        weights_path = explicit_weights
        if not os.path.isfile(weights_path):
            raise FileNotFoundError(f"Weights file not found: {weights_path}")
    else:
        best = os.path.join(model_dir, "best_model.pth")
        last = os.path.join(model_dir, "last_model.pth")
        if os.path.isfile(best):
            weights_path = best
        elif os.path.isfile(last):
            weights_path = last
        else:
            raise FileNotFoundError(f"No weights found in {model_dir} (expected best_model.pth or last_model.pth)")

    classes_json = os.path.join(model_dir, "classes.json")
    config_json = os.path.join(model_dir, "config.json")
    if not os.path.isfile(classes_json):
        classes_json = ""  # Will fallback to checkpoint['classes']
    if not os.path.isfile(config_json):
        config_json = ""   # Will fallback to checkpoint['config']

    return weights_path, classes_json, config_json


def load_classes(classes_json: str, checkpoint: dict) -> List[str]:
    # Prefer classes.json { "0": "cat", "1": "dog", ... }, else checkpoint['classes'] list
    if classes_json and os.path.isfile(classes_json):
        with open(classes_json, "r", encoding="utf-8") as f:
            mapping = json.load(f)
        # mapping keys may be str indices
        items = sorted(((int(k), v) for k, v in mapping.items()), key=lambda x: x[0])
        return [v for _, v in items]
    # fallback
    if isinstance(checkpoint, dict) and "classes" in checkpoint and isinstance(checkpoint["classes"], (list, tuple)):
        return list(checkpoint["classes"])
    raise RuntimeError("Unable to recover class labels (missing classes.json and checkpoint['classes']).")


def load_config(config_json: str, checkpoint: dict) -> dict:
    if config_json and os.path.isfile(config_json):
        with open(config_json, "r", encoding="utf-8") as f:
            return json.load(f)
    if isinstance(checkpoint, dict) and "config" in checkpoint and isinstance(checkpoint["config"], dict):
        return checkpoint["config"]
    return {}  # minimal fallback


def build_model(num_classes: int, cfg: dict, device: torch.device) -> CNNLSTM:
    hidden_size = int(cfg.get("hidden_size", 256))
    num_layers = int(cfg.get("num_layers", 2))
    bidirectional = bool(cfg.get("bidirectional", False))
    dropout = float(cfg.get("dropout", 0.3))
    freeze_cnn = bool(cfg.get("freeze_cnn", True))
    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=hidden_size,
        num_layers=num_layers,
        bidirectional=bidirectional,
        dropout=dropout,
        freeze_cnn=freeze_cnn
    ).to(device)
    model.eval()
    for p in model.parameters():
        p.requires_grad_(False)
    return model


def load_weights(model: CNNLSTM, weights_path: str, device: torch.device) -> None:
    ckpt = torch.load(weights_path, map_location=device)
    state = ckpt.get("model_state", ckpt)
    model.load_state_dict(state, strict=True)


def preprocess_frame(frame_bgr: np.ndarray, img_size: int, device: torch.device) -> torch.Tensor:
    # BGR -> RGB
    rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
    # Resize to model input size
    resized = cv2.resize(rgb, (img_size, img_size), interpolation=cv2.INTER_LINEAR)
    # To torch CHW float32 in [0,1]
    tensor = torch.from_numpy(resized).permute(2, 0, 1).float() / 255.0
    # Normalize
    tensor = (tensor - IMAGENET_MEAN) / IMAGENET_STD
    return tensor.to(device)


@torch.inference_mode()
def predict_from_window(model: CNNLSTM, window: torch.Tensor) -> torch.Tensor:
    """
    window: (T, 3, H, W) on device
    returns logits: (num_classes,)
    Implements the pipeline explicitly:
      frames -> model.cnn -> embeddings -> model.lstm -> classifier
    """
    T_, C, H, W = window.shape
    # Extract CNN features per frame
    feats = model.cnn(window)            # (T, 512, 1, 1)
    feats = feats.view(1, T_, -1)        # (1, T, 512)
    # LSTM over time
    lstm_out, _ = model.lstm(feats)      # (1, T, H_lstm)
    last = lstm_out[:, -1, :]            # (1, H_lstm)
    logits = model.classifier(last)[0]   # (num_classes,)
    return logits


def run_inference(
    source: str,
    weights: Optional[str],
    model_dir: Optional[str],
    display: bool,
    stride: int,
    warmup_pad: bool
) -> None:
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    torch.backends.cudnn.benchmark = True

    weights_path, classes_json, config_json = find_weights_and_assets(weights, model_dir)
    checkpoint = torch.load(weights_path, map_location=device)
    classes = load_classes(classes_json, checkpoint)
    cfg = load_config(config_json, checkpoint)
    img_size = int(cfg.get("img_size", 224))
    seq_len = int(cfg.get("sequence_length", 8))

    model = build_model(num_classes=len(classes), cfg=cfg, device=device)
    load_weights(model, weights_path, device)

    # Resolve source: webcam index or file path
    cap_source = 0
    try:
        # Accept integers like "0", "1"
        cap_source = int(source)
        is_camera = True
    except ValueError:
        cap_source = source
        is_camera = False

    cap = cv2.VideoCapture(cap_source)
    if not cap.isOpened():
        raise RuntimeError(f"Failed to open video source: {source}")

    frame_buffer: Deque[torch.Tensor] = deque(maxlen=seq_len)
    last_pred = ""
    fps_hist: Deque[float] = deque(maxlen=30)

    while True:
        ok, frame = cap.read()
        if not ok:
            break
        # Optional frame skipping
        for _ in range(max(0, stride - 1)):
            _ = cap.read()

        t0 = time.time()
        tensor = preprocess_frame(frame, img_size, device)
        frame_buffer.append(tensor)

        # Prepare window of length seq_len
        if len(frame_buffer) < seq_len:
            if not warmup_pad:
                pred_label = f"Warming up {len(frame_buffer)}/{seq_len}"
                conf = 0.0
            else:
                # Pad by repeating last frame to reach seq_len
                padded = list(frame_buffer) + [frame_buffer[-1]] * (seq_len - len(frame_buffer))
                window = torch.stack(padded, dim=0)  # (T,3,H,W)
                logits = predict_from_window(model, window)
                probs = F.softmax(logits, dim=-1)
                conf, idx = torch.max(probs, dim=-1)
                pred_label = classes[int(idx)]
                conf = float(conf.item())
        else:
            window = torch.stack(list(frame_buffer), dim=0)  # (T,3,H,W)
            logits = predict_from_window(model, window)
            probs = F.softmax(logits, dim=-1)
            conf, idx = torch.max(probs, dim=-1)
            pred_label = classes[int(idx)]
            conf = float(conf.item())

        last_pred = f"{pred_label} ({conf*100:.1f}%)"
        dt = time.time() - t0
        fps = 1.0 / max(dt, 1e-6)
        fps_hist.append(fps)
        avg_fps = sum(fps_hist) / max(1, len(fps_hist))

        if display:
            overlay = frame.copy()
            cv2.putText(overlay, f"Prediction: {last_pred}", (12, 28), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0, 255, 0), 2, cv2.LINE_AA)
            cv2.putText(overlay, f"FPS: {avg_fps:.1f}", (12, 56), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 0), 2, cv2.LINE_AA)
            cv2.imshow("Behavior Recognition (CNN+LSTM)", overlay)
            key = cv2.waitKey(1) & 0xFF
            if key in (27, ord('q')):  # ESC or 'q'
                break

    cap.release()
    if display:
        cv2.destroyAllWindows()


def main():
    parser = argparse.ArgumentParser(description="Real-time behavior recognition using CNN+LSTM.")
    parser.add_argument("--source", type=str, default="0", help="Video source: camera index (e.g., '0') or file path.")
    parser.add_argument("--weights", type=str, default=None, help="Path to .pth checkpoint. Defaults to best_model.pth in models/LSTM_behavior_recognition.")
    parser.add_argument("--model-dir", type=str, default=None, help="Directory containing weights and classes.json/config.json. Defaults to models/LSTM_behavior_recognition.")
    parser.add_argument("--no-display", action="store_true", help="Disable video display window.")
    parser.add_argument("--stride", type=int, default=1, help="Process every Nth frame for speed.")
    parser.add_argument("--no-warmup-pad", action="store_true", help="Do not pad initial frames; wait until buffer is full.")
    args = parser.parse_args()

    run_inference(
        source=args.source,
        weights=args.weights,
        model_dir=args.model_dir,
        display=not args.no_display,
        stride=max(1, args.stride),
        warmup_pad=not args.no_warmup_pad
    )


if __name__ == "__main__":
    main()
