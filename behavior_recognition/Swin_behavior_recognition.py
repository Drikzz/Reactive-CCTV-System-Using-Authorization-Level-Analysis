from __future__ import annotations
import os
import json
import time
from dataclasses import dataclass
from typing import Optional, Tuple

import torch
import torch.nn as nn
from torch.utils.data import DataLoader
import torchvision.models as models

# Optional Optuna
try:
    import optuna  # type: ignore
except Exception:
    optuna = None

# Reuse dataset, splits, and utilities from the LSTM pipeline
from .LSTM_behavior_recognition import (
    LSTMConfig as _BaseConfig,
    build_train_val_test_loaders,
    run_epoch,             # reused for quick eval where AMP is not critical
    format_secs,
)

# -------------------------- Configuration -------------------------- #
@dataclass
class VideoSwinConfig(_BaseConfig):
    # Swin/finetune-friendly defaults for T4 16GB
    learning_rate: float = 2e-4
    batch_size: int = 8
    dropout: float = 0.1
    freeze_cnn: bool = False
    num_epochs: int = 10
    # Mixed precision + stability
    use_amp: bool = True
    grad_clip_norm: float = 1.0
    # Scheduler
    use_cosine: bool = True
    cosine_eta_min: float = 1e-6
    # Resume
    resume: bool = True
    resume_from: Optional[str] = None  # if None, try save_dir/last_model.pth
    # Optuna toggles (reuse base) + early stopping patience (reuse base)
    use_optuna: bool = False
    optuna_trials: int = 10
    early_stop_patience: int = 5
    # Override cache root to separate from LSTM cache
    cache_root: str = os.path.join('datasets', 'cache', 'Swin')


# -------------------------- Model -------------------------- #
class VideoSwinClassifier(nn.Module):
    """
    Prefer a torchvision Video Swin Transformer. If not available, fallback to image Swin (swin_t)
    processing frames independently and averaging logits across time.
    """
    def __init__(self, num_classes: int, dropout: float = 0.1, freeze_cnn: bool = False):
        super().__init__()
        self.is_video = False
        self.backbone = None

        # Try to build a video Swin (torchvision.models.video.*)
        video_model_built = False
        try:
            from torchvision.models import video as vmodels  # type: ignore
            # Candidates: swin3d_t, video_swin_t (try multiple API variants for robustness)
            # 1) swin3d_t
            try:
                weights = vmodels.Swin3D_T_Weights.DEFAULT  # type: ignore[attr-defined]
                model = vmodels.swin3d_t(weights=weights)   # type: ignore[attr-defined]
                self._replace_head(model, num_classes, dropout)
                self.is_video = True
                self.backbone = model
                video_model_built = True
            except Exception:
                pass
            # 2) video_swin_t
            if not video_model_built:
                try:
                    weights = vmodels.VideoSwin_T_Weights.DEFAULT  # type: ignore[attr-defined]
                    model = vmodels.video_swin_t(weights=weights)   # type: ignore[attr-defined]
                    self._replace_head(model, num_classes, dropout)
                    self.is_video = True
                    self.backbone = model
                    video_model_built = True
                except Exception:
                    pass
        except Exception:
            video_model_built = False

        # Fallback: image Swin (frame-wise)
        if not video_model_built:
            try:
                weights = models.Swin_T_Weights.IMAGENET1K_V1  # type: ignore[attr-defined]
            except Exception:
                weights = None
            model = models.swin_t(weights=weights)
            in_features = model.head.in_features
            model.head = nn.Sequential(
                nn.Dropout(p=dropout),
                nn.Linear(in_features, num_classes),
            )
            self.backbone = model
            self.is_video = False

        # Freeze if requested (keep head trainable)
        if freeze_cnn:
            for p in self.backbone.parameters():
                p.requires_grad = False
            # Unfreeze classification head
            for m in self._iter_heads(self.backbone):
                for p in m.parameters():
                    p.requires_grad = True

    def _iter_heads(self, model):
        # Yield likely classification head modules across variants
        if hasattr(model, 'head') and isinstance(model.head, nn.Module):
            yield model.head
        if hasattr(model, 'fc') and isinstance(model.fc, nn.Module):
            yield model.fc
        if hasattr(model, 'classifier') and isinstance(model.classifier, nn.Module):
            yield model.classifier

    def _replace_head(self, model: nn.Module, num_classes: int, dropout: float):
        # Replace classification head with Dropout + Linear
        if hasattr(model, 'head') and isinstance(model.head, nn.Linear):
            in_features = model.head.in_features
            model.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        elif hasattr(model, 'head') and isinstance(model.head, nn.Module) and hasattr(model.head, 'in_features'):
            in_features = model.head.in_features  # type: ignore[attr-defined]
            model.head = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        elif hasattr(model, 'fc') and isinstance(model.fc, nn.Linear):
            in_features = model.fc.in_features
            model.fc = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        elif hasattr(model, 'classifier') and isinstance(model.classifier, nn.Linear):
            in_features = model.classifier.in_features
            model.classifier = nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes))
        else:
            # As a last resort, try to find a single Linear with out_features == num_classes and replace it.
            for name, m in model.named_modules():
                if isinstance(m, nn.Linear):
                    # replace this layer
                    parent = model
                    parts = name.split('.')
                    for p in parts[:-1]:
                        parent = getattr(parent, p)
                    in_features = m.in_features
                    setattr(parent, parts[-1], nn.Sequential(nn.Dropout(dropout), nn.Linear(in_features, num_classes)))
                    break

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        # x: (B, T, C, H, W)
        if self.is_video:
            x = x.permute(0, 2, 1, 3, 4).contiguous()  # (B, C, T, H, W)
            return self.backbone(x)
        # Image fallback: run per-frame then average logits
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        logits = self.backbone(x)  # (B*T, num_classes)
        logits = logits.view(B, T, -1).mean(dim=1)
        return logits


# -------------------------- Optuna Tuning (optional) -------------------------- #
def tune_with_optuna(cfg: VideoSwinConfig):
    if optuna is None:
        print("Optuna not installed. Run: pip install optuna")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Optuna tuning on device: {device}")

    def build_loaders_for_bs(bs: int):
        return build_train_val_test_loaders(cfg, batch_size_override=bs)

    def objective(trial: "optuna.trial.Trial") -> float:
        lr = trial.suggest_float("learning_rate", 1e-5, 5e-4, log=True)
        dropout = trial.suggest_float("dropout", 0.0, 0.6)
        bs_choices = sorted({cfg.batch_size, max(1, cfg.batch_size // 2), cfg.batch_size * 2})
        bs = trial.suggest_categorical("batch_size", bs_choices)

        train_loader, val_loader, _test_loader, classes = build_loaders_for_bs(bs)
        model = VideoSwinClassifier(num_classes=len(classes), dropout=dropout, freeze_cnn=cfg.freeze_cnn).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=lr, weight_decay=0.01)

        best_val_loss = float('inf')
        best_state = None
        epochs_no_improve = 0

        for epoch in range(cfg.num_epochs):
            # Training (simple, no amp in HPO for speed/robustness)
            model.train()
            running = 0.0
            total = 0
            for clips, labels in train_loader:
                clips, labels = clips.to(device), labels.to(device)
                optimizer.zero_grad()
                outputs = model(clips)
                loss = criterion(outputs, labels)
                loss.backward()
                optimizer.step()
                running += loss.item() * labels.size(0)
                total += labels.size(0)
            val_loss, _ = run_epoch(model, val_loader, criterion, None, device)

            if val_loss + 1e-8 < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.detach().cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= cfg.early_stop_patience:
                    break

            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        if best_state is not None:
            model.load_state_dict(best_state)
        return best_val_loss

    study = optuna.create_study(direction="minimize")
    n_trials = min(cfg.optuna_trials, 10)
    study.optimize(objective, n_trials=n_trials)

    print(f"Best trial: {study.best_trial.number}")
    print(f"Best params: {study.best_trial.params}")


# -------------------------- Training Loop with AMP, Cosine, Early Stopping -------------------------- #
def _save_checkpoint(path: str, model: nn.Module, optimizer, scheduler, scaler, epoch: int, best_val_loss: float, classes, cfg: VideoSwinConfig):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    torch.save({
        'model_state': {k: v.cpu() for k, v in model.state_dict().items()},
        'optimizer_state': optimizer.state_dict() if optimizer is not None else None,
        'scheduler_state': scheduler.state_dict() if scheduler is not None else None,
        'scaler_state': scaler.state_dict() if scaler is not None else None,
        'epoch': epoch,
        'best_val_loss': best_val_loss,
        'classes': classes,
        'config': cfg.__dict__,
    }, path)

def _try_resume(cfg: VideoSwinConfig, model: nn.Module, optimizer, scheduler, scaler, default_last_path: str) -> Tuple[int, float, Optional[list]]:
    start_epoch = 0
    best_val = float('inf')
    classes = None
    ckpt_path = cfg.resume_from if cfg.resume_from else default_last_path
    if cfg.resume and os.path.isfile(ckpt_path):
        print(f"Resuming from checkpoint: {ckpt_path}")
        ckpt = torch.load(ckpt_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'], strict=False)
        if optimizer is not None and ckpt.get('optimizer_state') is not None:
            optimizer.load_state_dict(ckpt['optimizer_state'])
        if scheduler is not None and ckpt.get('scheduler_state') is not None:
            scheduler.load_state_dict(ckpt['scheduler_state'])
        if scaler is not None and ckpt.get('scaler_state') is not None:
            scaler.load_state_dict(ckpt['scaler_state'])
        start_epoch = int(ckpt.get('epoch', 0))
        best_val = float(ckpt.get('best_val_loss', float('inf')))
        classes = ckpt.get('classes', None)
    return start_epoch, best_val, classes

def train_example():
    cfg = VideoSwinConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")

    if cfg.use_optuna:
        tune_with_optuna(cfg)
        return

    # Build loaders (reuse LSTM utilities)
    try:
        train_loader, val_loader, test_loader, classes = build_train_val_test_loaders(cfg)
    except Exception as e:
        print(f"Dataset build failed: {e}\nPopulate datasets/behavior_clips/category/class/*.mp4.")
        return

    num_classes = len(classes)
    print(f"Discovered classes: {classes} (n={num_classes})")

    model = VideoSwinClassifier(
        num_classes=num_classes,
        dropout=cfg.dropout,
        freeze_cnn=cfg.freeze_cnn
    ).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.AdamW(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate, weight_decay=0.01)
    scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(optimizer, T_max=max(1, cfg.num_epochs), eta_min=cfg.cosine_eta_min) if cfg.use_cosine else None
    scaler = torch.cuda.amp.GradScaler(enabled=(cfg.use_amp and device.type == 'cuda'))

    # Prepare save directory and config/classes
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = os.path.join('models', model_name)
    os.makedirs(save_dir, exist_ok=True)
    config_path = os.path.join(save_dir, 'config.json')
    if not os.path.isfile(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.__dict__, f, indent=2)
    classes_path = os.path.join(save_dir, 'classes.json')
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)

    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')

    # Optional pre-caching warm-up
    if cfg.pre_cache_before_training:
        print("Preprocessing: decoding videos and building cache (one pass)...")
        t0 = time.time()
        try:
            for ds in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
                _ = [ds[i] for i in range(len(ds))]
        except Exception as e:
            print(f"Warning during preprocessing pass: {e}")
        prep_secs = time.time() - t0
        print(f"Preprocessing completed in {format_secs(prep_secs)} ({prep_secs:.1f}s)")
        unreadable_all = []
        for ds in [train_loader.dataset, val_loader.dataset, test_loader.dataset]:
            unreadable_all.extend(ds.unreadable_videos)
        unreadable_all = list(dict.fromkeys(unreadable_all))
        if len(unreadable_all) > 0:
            print("Unreadable videos detected (replaced with black frames):")
            for p in unreadable_all:
                try:
                    rel = os.path.relpath(p, cfg.root_dir)
                except Exception:
                    rel = p
                print(f"  - {rel}")

    # Resume support
    start_epoch, best_val_loss, _maybe_classes = _try_resume(cfg, model, optimizer, scheduler, scaler, last_model_path)
    if start_epoch > 0:
        print(f"Resumed at epoch {start_epoch}/{cfg.num_epochs} with best_val_loss={best_val_loss:.4f}")

    # Training
    epoch_times = []
    epochs_no_improve = 0

    for epoch in range(start_epoch, cfg.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device, non_blocking=True), labels.to(device, non_blocking=True)
            optimizer.zero_grad(set_to_none=True)
            with torch.cuda.amp.autocast(enabled=(cfg.use_amp and device.type == 'cuda')):
                outputs = model(clips)
                loss = criterion(outputs, labels)
            if scaler.is_enabled():  # type: ignore[attr-defined]
                scaler.scale(loss).backward()
                if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                    scaler.unscale_(optimizer)
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                scaler.step(optimizer)
                scaler.update()
            else:
                loss.backward()
                if cfg.grad_clip_norm is not None and cfg.grad_clip_norm > 0:
                    torch.nn.utils.clip_grad_norm_(model.parameters(), cfg.grad_clip_norm)
                optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            pct = 100.0 * (batch_idx + 1) / max(1, total_batches)
            print(f"Epoch {epoch+1}/{cfg.num_epochs} | Batch {batch_idx+1}/{total_batches} ({pct:5.1f}%) - batch_loss={loss.item():.4f}", end='\r')

        # End of epoch
        print()
        train_loss = running_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)

        # Validation (no optimizer)
        val_loss, val_acc = run_epoch(model, val_loader, criterion, None, device)

        # Scheduler step (per-epoch)
        if scheduler is not None:
            scheduler.step()

        dur = time.time() - epoch_start
        epoch_times.append(dur)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = cfg.num_epochs - (epoch + 1)
        eta_total = remaining * avg_epoch

        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] "
              f"train_loss={train_loss:.4f} train_acc={train_acc:.2f}% | "
              f"val_loss={val_loss:.4f} val_acc={val_acc:.2f}% | time={format_secs(dur)} | ETA total={format_secs(eta_total)}")

        # Early stopping and checkpointing (best on val loss)
        improved = val_loss + 1e-8 < best_val_loss
        if improved:
            best_val_loss = val_loss
            epochs_no_improve = 0
            _save_checkpoint(best_model_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_loss, classes, cfg)
            print(f"  -> Saved new best model to {best_model_path}")
        else:
            epochs_no_improve += 1

        # Always save last snapshot (with states for resume)
        _save_checkpoint(last_model_path, model, optimizer, scheduler, scaler, epoch + 1, best_val_loss, classes, cfg)

        if epochs_no_improve >= cfg.early_stop_patience:
            print(f"Early stopping at epoch {epoch+1} (no improvement for {cfg.early_stop_patience} epochs).")
            break

    print("Training finished.")
    # Load best weights for final test evaluation
    if os.path.isfile(best_model_path):
        ckpt = torch.load(best_model_path, map_location='cpu')
        model.load_state_dict(ckpt['model_state'])
    test_loss, test_acc = run_epoch(model, test_loader, criterion, None, device)
    print(f"Test: loss={test_loss:.4f} acc={test_acc:.2f}% on {len(test_loader.dataset)} samples")
    print(f"Best model saved at: {best_model_path}")
    print(f"Last model snapshot saved at: {last_model_path}")


if __name__ == '__main__':
    train_example()
