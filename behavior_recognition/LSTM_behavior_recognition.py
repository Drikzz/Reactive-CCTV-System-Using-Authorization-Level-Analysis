"""LSTM-based dataset + dataloader setup for behavior recognition from MP4 clips.

Directory expectation (train/test split is done automatically):
    datasets/behavior_clips/
        ClassA/
            vid1.mp4
            vid2.mp4
        ClassB/
            vid3.mp4
            ...

If you only have a flat folder now, create subfolders per class first.

This script will:
 1. Discover classes (sub-folder names) under datasets/behavior_clips
 2. Split videos per class into train/test (default 80/20)
 3. Build PyTorch Dataset returning a sequence of frames (T, 3, H, W) + label
 4. Use a pretrained ResNet18 (frozen) as per-frame feature extractor feeding an LSTM
 5. Provide DataLoaders and example mini-training loop

Run directly:  python -m behavior_recognition.LSTM_behavior_recognition
"""
from __future__ import annotations
import os
import random
import math
import json
import time
from dataclasses import dataclass
from typing import List, Tuple, Optional, Callable, Set, Dict
from threading import Lock

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader
import torchvision.transforms as T
import torchvision.models as models

# -------------------------- Configuration -------------------------- #
@dataclass
class LSTMConfig:
    root_dir: str = os.path.join('datasets', 'behavior_clips')
    sequence_length: int = 8              # number of frames sampled per video / 16 default
    img_size: int = 224                    # resize shorter side then center crop / simple resize
    train_split: float = 0.8               # train/test split per class
    random_seed: int = 42
    batch_size: int = 4
    num_workers: int = 2                   # set >0 if you want parallel loading
    hidden_size: int = 256                 # LSTM hidden size
    num_layers: int = 2                    # LSTM layers
    bidirectional: bool = False
    dropout: float = 0.3                   # LSTM dropout (if num_layers>1)
    learning_rate: float = 1e-4
    num_epochs: int = 1
    freeze_cnn: bool = True                # freeze pretrained CNN weights
    max_videos_per_class: Optional[int] = None  # for quick debugging
    frame_sampling: str = 'stride'        # 'uniform' | 'stride' | 'random'
    show_video_conversion_progress: bool = True  # print % of videos converted to tensors
    cache_root: str = os.path.join('datasets', 'cache')  # root for cached tensors
    enable_cache: bool = True                            # toggle to disable caching
    pre_cache_before_training: bool = True               # run a full dataset pass to build cache and time it


# -------------------------- Dataset -------------------------- #

def format_secs(secs: float) -> str:
    secs = float(secs)
    if secs < 60:
        return f"{secs:.1f}s"
    minutes, s = divmod(int(round(secs)), 60)
    if minutes < 60:
        return f"{minutes}m{s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m{s:02d}s"
class BehaviorVideoDataset(Dataset):
    def __init__(
        self,
        video_label_pairs: List[Tuple[str, int]],
        classes: List[str],
        sequence_length: int,
        transform: Optional[Callable] = None,
        img_size: int = 224,
        frame_sampling: str = 'uniform',
        show_progress: bool = True,
        cache_root: Optional[str] = None,
        enable_cache: bool = True,
        dataset_root: Optional[str] = None,
    ):
        self.video_label_pairs = video_label_pairs
        self.classes = classes
        self.sequence_length = sequence_length
        self.img_size = img_size  # store for caching / fallback frame
        self.transform = transform or T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        self.frame_sampling = frame_sampling
        self.show_progress = show_progress
        # Tracking conversion progress (only counts first-time tensorization of a video)
        self._processed_videos = set()
        self._processed_count = 0
        self._progress_lock = Lock()
        self._total_videos = len(video_label_pairs)
        # Caching setup
        self.enable_cache = enable_cache and (cache_root is not None)
        self.cache_root = cache_root
        self.dataset_root = dataset_root
        if self.enable_cache:
            os.makedirs(self.cache_root, exist_ok=True)

    def __len__(self):
        return len(self.video_label_pairs)

    def _sample_indices(self, total_frames: int) -> List[int]:
        Tseq = self.sequence_length
        if total_frames <= 0:
            return [0] * Tseq
        if self.frame_sampling == 'random':
            if total_frames >= Tseq:
                return sorted(random.sample(range(total_frames), Tseq))
            else:
                base = list(range(total_frames))
                while len(base) < Tseq:
                    base.extend(base[: max(0, Tseq - len(base))])
                return base[:Tseq]
        elif self.frame_sampling == 'stride':
            stride = max(1, total_frames // Tseq)
            idxs = list(range(0, stride * Tseq, stride))
            if len(idxs) < Tseq:
                idxs.extend([total_frames - 1] * (Tseq - len(idxs)))
            return [min(i, total_frames - 1) for i in idxs]
        else:  # uniform
            if total_frames >= Tseq:
                return [int(round(x)) for x in np.linspace(0, total_frames - 1, Tseq)]
            else:
                idxs = list(range(total_frames))
                while len(idxs) < Tseq:
                    idxs.extend(idxs[: max(0, Tseq - len(idxs))])
                return idxs[:Tseq]

    def __getitem__(self, idx):
        video_path, label = self.video_label_pairs[idx]
        # ---- Determine cache path first ----
        cache_hit = False
        cache_path: Optional[str] = None
        clip: Optional[torch.Tensor] = None
        if self.enable_cache:
            # derive relative path preserving class subfolder
            if self.dataset_root and os.path.commonpath([os.path.abspath(video_path), os.path.abspath(self.dataset_root)]) == os.path.abspath(self.dataset_root):
                rel = os.path.relpath(video_path, self.dataset_root)
            else:
                rel = os.path.basename(video_path)
            rel_no_ext, _ = os.path.splitext(rel)
            # normalize and sanitize
            rel_no_ext = rel_no_ext.replace('\\', '/').replace('..', '')
            rel_no_ext_base = os.path.basename(rel_no_ext)
            size_tag = self.img_size
            cache_filename = f"{rel_no_ext_base}_T{self.sequence_length}_S{self.frame_sampling}_IMG{size_tag}.pt"
            subdir = os.path.dirname(rel)
            cache_dir = os.path.join(self.cache_root, subdir)
            os.makedirs(cache_dir, exist_ok=True)
            cache_path = os.path.join(cache_dir, cache_filename)
            if os.path.isfile(cache_path):
                try:
                    clip = torch.load(cache_path, map_location='cpu')
                    if isinstance(clip, torch.Tensor) and clip.ndim == 4 and clip.shape[0] == self.sequence_length:
                        cache_hit = True
                except Exception:
                    cache_hit = False
        if not cache_hit:
            cap = cv2.VideoCapture(video_path)
            if not cap.isOpened():
                raise RuntimeError(f"Failed to open video: {video_path}")
            frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
            indices = self._sample_indices(frame_count)
            wanted = set(indices)
            cur = 0
            grabbed_frames = {}
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                if cur in wanted:
                    frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                    pil_like = T.functional.to_pil_image(frame_rgb)
                    tensor = self.transform(pil_like)
                    grabbed_frames[cur] = tensor
                    if len(grabbed_frames) == len(wanted):
                        break
                cur += 1
            cap.release()
            ordered = [grabbed_frames.get(i) for i in indices]
            if ordered[0] is None:
                # fallback black frame
                fallback = torch.zeros(3, self.img_size, self.img_size)
                ordered[0] = fallback
            for i in range(1, len(ordered)):
                if ordered[i] is None:
                    ordered[i] = ordered[i - 1]
            clip = torch.stack(ordered, dim=0)
            if self.enable_cache and cache_path is not None:
                try:
                    torch.save(clip.cpu(), cache_path)
                except Exception as e:
                    if self.show_progress:
                        print(f"Warning: failed to cache {cache_path}: {e}")
        # At this point clip must be a tensor
        assert clip is not None, "Internal error: clip tensor not created"
        # Progress reporting (only first time per video)
        if self.show_progress:
            with self._progress_lock:
                if video_path not in self._processed_videos:
                    self._processed_videos.add(video_path)
                    self._processed_count += 1
                    pct = 100.0 * self._processed_count / max(1, self._total_videos)
                    origin = 'CACHE' if cache_hit else 'DECODE'
                    print(f"Video tensor ready {self._processed_count}/{self._total_videos} ({pct:5.1f}%) [{origin}] - {os.path.basename(video_path)}")
        return clip, label


# -------------------------- Model -------------------------- #
class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, num_layers: int = 2, bidirectional: bool = False, dropout: float = 0.3, freeze_cnn: bool = True):
        super().__init__()
        backbone = models.resnet18(weights=models.ResNet18_Weights.DEFAULT)
        # remove fc
        self.cnn = nn.Sequential(*list(backbone.children())[:-1])  # output (B,512,1,1)
        if freeze_cnn:
            for p in self.cnn.parameters():
                p.requires_grad = False
        feature_dim = 512
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_size,
            num_layers=num_layers,
            batch_first=True,
            dropout=dropout if num_layers > 1 else 0.0,
            bidirectional=bidirectional,
        )
        lstm_out_dim = hidden_size * (2 if bidirectional else 1)
        self.classifier = nn.Sequential(
            nn.Linear(lstm_out_dim, lstm_out_dim // 2),
            nn.ReLU(inplace=True),
            nn.Dropout(0.3),
            nn.Linear(lstm_out_dim // 2, num_classes)
        )

    def forward(self, x: torch.Tensor):  # x: (B,T,C,H,W)
        B, T, C, H, W = x.shape
        x = x.view(B * T, C, H, W)
        feats = self.cnn(x)  # (B*T,512,1,1)
        feats = feats.view(B, T, -1)  # (B,T,512)
        lstm_out, _ = self.lstm(feats)  # (B,T,Hid)
        # Use last time-step
        last = lstm_out[:, -1, :]
        logits = self.classifier(last)
        return logits


# -------------------------- Utility Functions -------------------------- #

def discover_videos(root_dir: str, max_videos_per_class: Optional[int] = None) -> Tuple[List[Tuple[str, int]], List[str]]:
    classes = []
    video_label_pairs: List[Tuple[str, int]] = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    for entry in sorted(os.listdir(root_dir)):
        class_path = os.path.join(root_dir, entry)
        if os.path.isdir(class_path):
            classes.append(entry)
            vids = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith('.mp4')]
            vids = sorted(vids)
            if max_videos_per_class is not None:
                vids = vids[:max_videos_per_class]
            label = len(classes) - 1
            for v in vids:
                video_label_pairs.append((v, label))
    if not classes:
        raise RuntimeError(f"No class subdirectories with videos found under {root_dir}. Create structure root_dir/ClassName/*.mp4")
    return video_label_pairs, classes


def make_splits(video_label_pairs: List[Tuple[str, int]], classes: List[str], train_split: float, seed: int) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]:
    random.seed(seed)
    by_class = {i: [] for i in range(len(classes))}
    for path, label in video_label_pairs:
        by_class[label].append(path)
    train_pairs: List[Tuple[str,int]] = []
    test_pairs: List[Tuple[str,int]] = []
    for label, paths in by_class.items():
        random.shuffle(paths)
        k = int(math.ceil(len(paths) * train_split))
        train_paths = paths[:k]
        test_paths = paths[k:] if k < len(paths) else []
        train_pairs.extend([(p, label) for p in train_paths])
        test_pairs.extend([(p, label) for p in test_paths])
    return train_pairs, test_pairs


# -------------------------- Build Datasets & Loaders -------------------------- #

def build_dataloaders(cfg: LSTMConfig):
    video_label_pairs, classes = discover_videos(cfg.root_dir, cfg.max_videos_per_class)
    train_pairs, test_pairs = make_splits(video_label_pairs, classes, cfg.train_split, cfg.random_seed)
    transform = T.Compose([
        T.ToTensor(),
        T.Resize((cfg.img_size, cfg.img_size)),
        T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    ])
    train_ds = BehaviorVideoDataset(
        train_pairs, classes, cfg.sequence_length, transform, cfg.img_size,
        cfg.frame_sampling, show_progress=cfg.show_video_conversion_progress,
        cache_root=cfg.cache_root, enable_cache=cfg.enable_cache, dataset_root=cfg.root_dir
    )
    test_ds = BehaviorVideoDataset(
        test_pairs, classes, cfg.sequence_length, transform, cfg.img_size,
        cfg.frame_sampling, show_progress=False,
        cache_root=cfg.cache_root, enable_cache=cfg.enable_cache, dataset_root=cfg.root_dir
    )
    train_loader = DataLoader(train_ds, batch_size=cfg.batch_size, shuffle=True, num_workers=cfg.num_workers, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=cfg.batch_size, shuffle=False, num_workers=cfg.num_workers, pin_memory=True)
    return train_loader, test_loader, classes


# -------------------------- Example Training Loop -------------------------- #

def train_example():
    cfg = LSTMConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Using device: {device}")
    try:
        train_loader, test_loader, classes = build_dataloaders(cfg)
    except Exception as e:
        print(f"Dataset build failed: {e}\nPopulate datasets/behavior_clips/ with class subfolders and mp4 files.")
        return
    num_classes = len(classes)
    print(f"Discovered classes: {classes} (n={num_classes})")

    model = CNNLSTM(num_classes=num_classes, hidden_size=cfg.hidden_size, num_layers=cfg.num_layers, bidirectional=cfg.bidirectional, dropout=cfg.dropout, freeze_cnn=cfg.freeze_cnn).to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)

    # Prepare model save directory
    model_name = os.path.splitext(os.path.basename(__file__))[0]  # file name without extension
    save_dir = os.path.join('models', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Persist config once
    config_path = os.path.join(save_dir, 'config.json')
    if not os.path.isfile(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.__dict__, f, indent=2)

    # Save classes mapping
    classes_path = os.path.join(save_dir, 'classes.json')
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)

    best_acc = -1.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')

    # -------------------------- Preprocessing Timing -------------------------- #
    # Warm-up/cache build timing: iterate through train and test datasets once to force
    # decoding and caching, then report total preprocessing time.
    if cfg.pre_cache_before_training:
        print("Preprocessing: decoding videos and building cache (one pass)...")
        t0 = time.time()
        # Iterate dataset objects rather than loaders to avoid batching overhead
        # Use a no-grad context for safety (though dataset __getitem__ doesn't use autograd)
        try:
            _ = [train_loader.dataset[i] for i in range(len(train_loader.dataset))]
            _ = [test_loader.dataset[i] for i in range(len(test_loader.dataset))]
        except Exception as e:
            print(f"Warning during preprocessing pass: {e}")
        t1 = time.time()
        prep_secs = t1 - t0
        print(f"Preprocessing completed in {format_secs(prep_secs)} ({prep_secs:.1f}s)")

    epoch_times: List[float] = []
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)
        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips = clips.to(device)
            labels = labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            # Progress percentage for current epoch
            pct = 100.0 * (batch_idx + 1) / total_batches
            print(f"Epoch {epoch+1}/{cfg.num_epochs} | Batch {batch_idx+1}/{total_batches} ({pct:5.1f}%) - batch_loss={loss.item():.4f}", end='\r')
        # After epoch, ensure newline
        print()
        train_loss = running_loss / max(1,total)
        train_acc = 100.0 * correct / max(1,total)
        epoch_dur = time.time() - epoch_start
        epoch_times.append(epoch_dur)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = cfg.num_epochs - (epoch + 1)
        eta_total = remaining * avg_epoch
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] loss={train_loss:.4f} acc={train_acc:.2f}% | time={format_secs(epoch_dur)} | ETA total={format_secs(eta_total)}")
        # one quick evaluation pass (optional)
        if (epoch+1) % 5 == 0 or epoch == cfg.num_epochs - 1:
            model.eval()
            correct_eval = 0
            total_eval = 0
            with torch.no_grad():
                for clips, labels in test_loader:
                    clips, labels = clips.to(device), labels.to(device)
                    outputs = model(clips)
                    _, pred = outputs.max(1)
                    total_eval += labels.size(0)
                    correct_eval += pred.eq(labels).sum().item()
            test_acc = 100.0 * correct_eval / max(1,total_eval)
            print(f"  Test Acc: {test_acc:.2f}% (on {total_eval} samples)")
            # Save best model
            if test_acc > best_acc:
                best_acc = test_acc
                torch.save({
                    'model_state': model.state_dict(),
                    'epoch': epoch+1,
                    'test_acc': best_acc,
                    'classes': classes,
                    'config': cfg.__dict__,
                }, best_model_path)
                print(f"  -> Saved new best model to {best_model_path}")

        # Always save last epoch snapshot
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'classes': classes,
            'config': cfg.__dict__,
        }, last_model_path)
        # Optional per-epoch named checkpoint (comment out if too many files)
        # torch.save(model.state_dict(), os.path.join(save_dir, f'epoch_{epoch+1:03d}.pth'))

    print("Training example finished.")
    if os.path.isfile(best_model_path):
        print(f"Best model saved at: {best_model_path} (acc={best_acc:.2f}%)")
    print(f"Last model snapshot saved at: {last_model_path}")


if __name__ == '__main__':
    train_example()
