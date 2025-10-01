"""MobileNetV2 + LSTM behavior recognition training script.

This file mirrors the functionality of `lstm_resnet/LSTM_behavior_recognition.py` but
uses a MobileNetV2 2D CNN backbone for feature extraction instead of ResNet18.

Key points:
- Keeps identical dataset discovery, caching, augmentation, and sampling logic.
- Feature dimension from MobileNetV2 is 1280 (final conv output) -> projected (optional) -> LSTM.
- Configuration dataclass duplicated with name MobilenetLSTMConfig to avoid confusion.
- Saves under models/LSTM_behavior_mobilenet/ (directory name derived from this filename).
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
import gc

import cv2
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models

# -------------------------- Configuration -------------------------- #
@dataclass
class MobilenetLSTMConfig:
    root_dir: str = os.path.join('datasets', 'behavior_clips')
    cache_root: str = os.path.join('datasets', 'cache', 'LSTM_MN')
    sequence_length: int = 32
    img_size: int = 224
    frame_sampling: str = 'uniform'
    train_split: float = 0.8
    val_split: float = 0.2
    random_seed: int = 42
    batch_size: int = 16
    num_workers: int = 2
    max_videos_per_class: Optional[int] = 200
    hidden_size: int = 512
    num_layers: int = 3
    bidirectional: bool = False
    dropout: float = 0.3
    freeze_cnn: bool = True
    learning_rate: float = 1e-4
    num_epochs: int = 50
    early_stop_patience: int = 5
    enable_cache: bool = True
    pre_cache_before_training: bool = True
    show_video_conversion_progress: bool = True
    cache_format: str = 'pt'
    aug_flip_p: float = 0.5
    aug_crop_scale: Tuple[float,float] = (0.8, 1.0)
    aug_crop_ratio: Tuple[float,float] = (0.9, 1.1)
    aug_brightness: Tuple[float,float] = (0.8, 1.2)
    aug_contrast: Tuple[float,float] = (0.8, 1.2)
    use_optuna: bool = False
    optuna_trials: int = 10

# (We reuse dataset, discovery, and helper components by importing from the resnet script if available)
# To avoid tight coupling, we replicate minimal necessary logic here.

# -------------------------- Helper Functions -------------------------- #

def format_secs(secs: float) -> str:
    secs = float(secs)
    if secs < 60:
        return f"{secs:.1f}s"
    minutes, s = divmod(int(round(secs)), 60)
    if minutes < 60:
        return f"{minutes}m{s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m{s:02d}s"

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def seed_worker(worker_id: int):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))


def _abs_under_root(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_project_root(), p))

# -------------------------- Dataset (simplified copy) -------------------------- #
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
        aug_flip_p: float = 0.5,
        aug_crop_scale: Tuple[float,float] = (0.8, 1.0),
        aug_crop_ratio: Tuple[float,float] = (0.9, 1.1),
        aug_brightness: Tuple[float,float] = (0.8, 1.2),
        aug_contrast: Tuple[float,float] = (0.8, 1.2),
        cache_format: str = 'pt',
    ):
        self.classes = classes
        self.sequence_length = sequence_length
        self.img_size = img_size
        self._preprocess = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
        ])
        if isinstance(transform, T.Normalize):
            mean = transform.mean
            std = transform.std
        else:
            mean = [0.485, 0.456, 0.406]
            std = [0.229, 0.224, 0.225]
        self._mean = torch.tensor(mean, dtype=torch.float32).view(1, 3, 1, 1)
        self._std = torch.tensor(std, dtype=torch.float32).view(1, 3, 1, 1)
        self.frame_sampling = frame_sampling
        self.show_progress = show_progress
        self._processed_videos = set()
        self._processed_count = 0
        self._progress_lock = Lock()
        self._total_videos = len(video_label_pairs)
        self.enable_cache = enable_cache and (cache_root is not None)
        self.cache_root = cache_root
        self.dataset_root = dataset_root
        self.cache_ext = '.pt' if str(cache_format).lower() == 'pt' else '.npy'
        if self.enable_cache and self.cache_root:
            os.makedirs(self.cache_root, exist_ok=True)
        self.unreadable_videos: List[str] = []
        self._unreadable_set: Set[str] = set()
        self._unreadable_lock = Lock()
        self.paths: List[str] = []
        self.labels_only: List[int] = []
        self.aug_flags: List[bool] = []
        self.original_counts: Dict[int, int] = {i: 0 for i in range(len(classes))}
        self.augmented_counts: Dict[int, int] = {i: 0 for i in range(len(classes))}
        for item in video_label_pairs:
            if len(item) == 2:
                pth, lab = item
                augf = False
            else:
                pth, lab, augf = item  # type: ignore
            self.paths.append(pth)
            self.labels_only.append(int(lab))
            self.aug_flags.append(bool(augf))
            if 0 <= int(lab) < len(classes):
                if bool(augf):
                    self.augmented_counts[int(lab)] += 1
                else:
                    self.original_counts[int(lab)] += 1
        n_classes = len(classes)
        self.class_counts: List[int] = [0] * n_classes
        for lab in self.labels_only:
            if 0 <= lab < n_classes:
                self.class_counts[lab] += 1
        self.aug_flip_p = aug_flip_p
        self.aug_crop_scale = aug_crop_scale
        self.aug_crop_ratio = aug_crop_ratio
        self.aug_brightness = aug_brightness
        self.aug_contrast = aug_contrast

    def __len__(self):
        return len(self.paths)

    def _sample_indices(self, total_frames: int) -> List[int]:
        Tseq = self.sequence_length
        if total_frames <= 0:
            return [0] * Tseq
        if self.frame_sampling == 'random':
            if total_frames >= Tseq:
                return sorted(random.sample(range(total_frames), Tseq))
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
            idxs = list(range(total_frames))
            while len(idxs) < Tseq:
                idxs.extend(idxs[: max(0, Tseq - len(idxs))])
            return idxs[:Tseq]

    def _cache_path_for(self, video_path: str) -> Tuple[str, bool]:
        if self.cache_root is None:
            return "", False
        lp = video_path.lower()
        if lp.endswith('.npy') or lp.endswith('.pt'):
            return video_path, True
        if self.dataset_root and os.path.commonpath([os.path.abspath(video_path), os.path.abspath(self.dataset_root)]) == os.path.abspath(self.dataset_root):
            rel = os.path.relpath(video_path, self.dataset_root)
        else:
            rel = os.path.basename(video_path)
        rel_no_ext, _ = os.path.splitext(rel)
        rel_no_ext = rel_no_ext.replace('\\', '/').replace('..', '')
        rel_no_ext_base = os.path.basename(rel_no_ext)
        cache_filename = f"{rel_no_ext_base}_T{self.sequence_length}_S{self.frame_sampling}_IMG{self.img_size}{self.cache_ext}"
        subdir = os.path.dirname(rel_no_ext)
        cache_dir = os.path.join(self.cache_root, subdir)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, cache_filename), False

    def _normalize_clip(self, clip: torch.Tensor) -> torch.Tensor:
        return (clip - self._mean) / self._std

    def _augment_clip_inplace(self, clip: torch.Tensor) -> torch.Tensor:
        Tlen, C, H, W = clip.shape
        do_flip = random.random() < self.aug_flip_p
        pil0 = T.functional.to_pil_image(clip[0].clamp(0,1))
        i, j, h, w = T.RandomResizedCrop.get_params(
            pil0, scale=self.aug_crop_scale, ratio=self.aug_crop_ratio
        )
        b_factor = random.uniform(*self.aug_brightness)
        c_factor = random.uniform(*self.aug_contrast)
        out_frames = []
        for t in range(Tlen):
            fr = clip[t]
            if do_flip:
                fr = torch.flip(fr, dims=[2])
            fr = T.functional.resized_crop(fr, i, j, h, w, (self.img_size, self.img_size), interpolation=T.InterpolationMode.BILINEAR)
            fr = T.functional.adjust_brightness(fr, b_factor)
            fr = T.functional.adjust_contrast(fr, c_factor)
            out_frames.append(fr)
        return torch.stack(out_frames, dim=0)

    def __getitem__(self, idx):
        path = self.paths[idx]
        label = self.labels_only[idx]
        do_aug = self.aug_flags[idx]

        clip_pre = None
        cache_path = ""
        # ---- Try cache load ----
        if self.enable_cache and self.cache_root:
            cache_path, _ = self._cache_path_for(path)

            def _to_tensor(obj):
                if isinstance(obj, torch.Tensor):
                    return obj.float()
                if isinstance(obj, np.ndarray):
                    return torch.from_numpy(obj).float()
                return None

            if os.path.isfile(cache_path):
                try:
                    if cache_path.endswith('.npy'):
                        arr = np.load(cache_path, allow_pickle=False, mmap_mode='r')
                        clip_pre = _to_tensor(np.asarray(arr))
                    else:
                        obj = torch.load(cache_path, map_location='cpu')
                        clip_pre = _to_tensor(obj)
                except Exception:
                    clip_pre = None

            # If loaded single frame or wrong shape, adjust
            if isinstance(clip_pre, torch.Tensor):
                if clip_pre.dim() == 3:  # (C,H,W) -> tile
                    clip_pre = clip_pre.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)
                elif clip_pre.dim() == 5:  # (N,T,C,H,W) -> choose deterministic clip
                    sel = (hash(path) % max(1, clip_pre.size(0)))
                    clip_pre = clip_pre[sel]

        # ---- Decode if cache miss ----
        if clip_pre is None:
            cap = cv2.VideoCapture(path)
            if not cap.isOpened():
                clip_pre = torch.zeros(self.sequence_length, 3, self.img_size, self.img_size)
            else:
                frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                indices = self._sample_indices(frame_count)
                wanted = set(indices)
                cur = 0
                grabbed: Dict[int, torch.Tensor] = {}
                while True:
                    ret, frame = cap.read()
                    if not ret:
                        break
                    if cur in wanted:
                        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                        pil_img = T.functional.to_pil_image(frame_rgb)
                        tensor = self._preprocess(pil_img)
                        grabbed[cur] = tensor
                        if len(grabbed) == len(wanted):
                            break
                    cur += 1
                cap.release()
                ordered = [grabbed.get(i) for i in indices]
                if ordered[0] is None:
                    ordered[0] = torch.zeros(3, self.img_size, self.img_size)
                for i in range(1, len(ordered)):
                    if ordered[i] is None:
                        ordered[i] = ordered[i-1]
                clip_pre = torch.stack(ordered, dim=0)

            # ---- Save to cache ----
            if self.enable_cache and self.cache_root and cache_path:
                try:
                    if self.cache_ext == '.pt':
                        torch.save(clip_pre.cpu(), cache_path)
                    else:
                        np.save(cache_path, clip_pre.cpu().numpy())
                except Exception:
                    pass  # silent fail

        # ---- Augment & Normalize ----
        if do_aug:
            clip_aug = self._augment_clip_inplace(clip_pre)
            clip = self._normalize_clip(clip_aug)
        else:
            clip = self._normalize_clip(clip_pre)
        return clip, label

# -------------------------- Discovery / Splits -------------------------- #

def discover_videos(root_dir: str, max_videos_per_class: Optional[int] = None) -> Tuple[List[Tuple[str, int]], List[str]]:
    classes: List[str] = []
    video_label_pairs: List[Tuple[str,int]] = []
    if not os.path.isdir(root_dir):
        raise FileNotFoundError(f"Root directory not found: {root_dir}")
    for category in sorted(os.listdir(root_dir)):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            for class_name in sorted(os.listdir(category_path)):
                class_path = os.path.join(category_path, class_name)
                if os.path.isdir(class_path):
                    full = f"{category}/{class_name}"
                    classes.append(full)
                    vids = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith('.mp4')]
                    vids = sorted(vids)
                    if max_videos_per_class is not None:
                        vids = vids[:max_videos_per_class]
                    label = len(classes) - 1
                    for v in vids:
                        video_label_pairs.append((v, label))
    if not classes:
        raise RuntimeError(f"No videos found under {root_dir}")
    return video_label_pairs, classes


def make_splits(pairs: List[Tuple[str,int]], classes: List[str], train_split: float, seed: int):
    random.seed(seed)
    by_class = {i: [] for i in range(len(classes))}
    for p, lab in pairs:
        by_class[lab].append(p)
    train_pairs: List[Tuple[str,int]] = []
    test_pairs: List[Tuple[str,int]] = []
    for lab, paths in by_class.items():
        random.shuffle(paths)
        k = int(math.ceil(len(paths) * train_split))
        tr = paths[:k]
        te = paths[k:]
        train_pairs.extend([(p, lab) for p in tr])
        test_pairs.extend([(p, lab) for p in te])
    return train_pairs, test_pairs

# ---- NEW: cap & augment helper (mirrors ResNet script logic) ----
def cap_and_augment_to_target(
    train_pairs: List[Tuple[str,int]],
    classes: List[str],
    target_per_class: int,
    seed: int
) -> List[Tuple[str,int,bool]]:
    rng = random.Random(seed)
    by_class: Dict[int, List[str]] = {i: [] for i in range(len(classes))}
    for p, lab in train_pairs:
        by_class[lab].append(p)
    out: List[Tuple[str,int,bool]] = []
    for lab, paths in by_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        if len(paths) == 0:
            continue
        if len(paths) > target_per_class:
            keep = paths[:target_per_class]
            out.extend([(p, lab, False) for p in keep])
        else:
            out.extend([(p, lab, False) for p in paths])
            need = target_per_class - len(paths)
            i = 0
            while i < need and len(paths) > 0:
                out.append((paths[i % len(paths)], lab, True))
                i += 1
    return out

# -------------------------- Model -------------------------- #
class MobileNetLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 512, num_layers: int = 3, bidirectional: bool = False, dropout: float = 0.3, freeze_cnn: bool = True):
        super().__init__()
        backbone = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.DEFAULT)
        # Extract features (final conv output is (B,1280,7,7)); we global average pool to (B,1280)
        self.cnn_features = backbone.features
        self.pool = nn.AdaptiveAvgPool2d((1,1))
        if freeze_cnn:
            for p in self.cnn_features.parameters():
                p.requires_grad = False
        feature_dim = 1280
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
        x = x.view(B*T, C, H, W)
        feats = self.cnn_features(x)
        feats = self.pool(feats)  # (B*T,1280,1,1)
        feats = feats.view(B, T, -1)  # (B,T,1280)
        lstm_out, _ = self.lstm(feats)
        last = lstm_out[:, -1, :]
        logits = self.classifier(last)
        return logits

# -------------------------- Dataloaders -------------------------- #

def build_dataloaders(cfg: MobilenetLSTMConfig):
    abs_root = _abs_under_root(cfg.root_dir)
    abs_cache = _abs_under_root(cfg.cache_root)
    pairs, classes = discover_videos(abs_root, cfg.max_videos_per_class)
    train_pairs, test_pairs = make_splits(pairs, classes, cfg.train_split, cfg.random_seed)

    # ---- NEW: balance classes to target with augmentation flags ----
    if cfg.max_videos_per_class is not None:
        train_pairs_aug = cap_and_augment_to_target(
            train_pairs, classes, cfg.max_videos_per_class, cfg.random_seed
        )
    else:
        train_pairs_aug = train_pairs  # no augmentation list

    # Progress summary of class distribution before/after
    counts_before: Dict[int,int] = {i:0 for i in range(len(classes))}
    for _p, lab in train_pairs:
        counts_before[lab] += 1
    counts_after: Dict[int,int] = {i:0 for i in range(len(classes))}
    for item in train_pairs_aug:
        lab = item[1]
        counts_after[lab] += 1
    print("Class balance (before -> after):")
    for i, cls in enumerate(classes):
        print(f"  {cls}: {counts_before[i]} -> {counts_after[i]} (target={cfg.max_videos_per_class})")

    normalize_only = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])
    set_global_seed(cfg.random_seed)
    g = torch.Generator().manual_seed(cfg.random_seed)

    train_ds = BehaviorVideoDataset(
        train_pairs_aug, classes, cfg.sequence_length, normalize_only, cfg.img_size, cfg.frame_sampling,
        cache_root=abs_cache, enable_cache=cfg.enable_cache, dataset_root=abs_root, cache_format=cfg.cache_format
    )
    test_ds = BehaviorVideoDataset(
        test_pairs, classes, cfg.sequence_length, normalize_only, cfg.img_size, cfg.frame_sampling,
        show_progress=False, cache_root=abs_cache, enable_cache=cfg.enable_cache, dataset_root=abs_root, cache_format=cfg.cache_format
    )

    class_counts = train_ds.class_counts
    sample_weights = [1.0 / max(1, class_counts[lab]) for lab in train_ds.labels_only]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True
    )
    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, classes

# -------------------------- Training -------------------------- #

def train_mobilenet_lstm():
    cfg = MobilenetLSTMConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seed(cfg.random_seed)
    print(f"Using device: {device}")

    try:
        train_loader, test_loader, classes = build_dataloaders(cfg)
    except Exception as e:
        print(f"Dataset build failed: {e}")
        return

    num_classes = len(classes)
    print(f"Classes (n={num_classes}): {classes}")
    model = MobileNetLSTM(
        num_classes=num_classes,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
        freeze_cnn=cfg.freeze_cnn
    ).to(device)

    class_counts = getattr(train_loader.dataset, 'class_counts', [0]*num_classes)
    class_weights = torch.tensor([1.0 / max(1,c) for c in class_counts], dtype=torch.float, device=device)
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)

    # Save directory
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = os.path.join('models', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save config & classes
    config_path = os.path.join(save_dir, 'config.json')
    if not os.path.isfile(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.__dict__, f, indent=2)
    with open(os.path.join(save_dir, 'classes.json'), 'w', encoding='utf-8') as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)

    best_acc = -1.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')

    # Optional preprocessing pass to build/cache clips (with progress & ETA)
    if getattr(cfg, 'pre_cache_before_training', False):
        print("Preprocessing dataset (train + test) to warm up / cache clips...")
        precache_start = time.time()
        datasets_seq = [('train', train_loader.dataset), ('test', test_loader.dataset)]
        total_items = sum(len(ds) for _name, ds in datasets_seq)
        processed = 0
        report_every = max(1, total_items // 100)
        for name, ds in datasets_seq:
            for i in range(len(ds)):
                _ = ds[i]
                processed += 1
                if processed % report_every == 0 or processed == total_items:
                    elapsed = time.time() - precache_start
                    rate = processed / elapsed if elapsed > 0 else 0.0
                    remaining = total_items - processed
                    eta = remaining / rate if rate > 0 else 0.0
                    pct = 100.0 * processed / max(1, total_items)
                    print(f"[Preprocess {name}] {processed}/{total_items} ({pct:5.1f}%) elapsed={elapsed:6.1f}s ETA={eta:6.1f}s rate={rate:5.2f} it/s", end='\r')
        print()
        total_precache_time = time.time() - precache_start
        print(f"Preprocessing complete in {total_precache_time:.1f}s")

    # ---- NEW: helper to save class counts each epoch ----
    def save_class_counts(epoch: int):
        counts_path = os.path.join(save_dir, f'class_counts_epoch_{epoch}.txt')
        ds = train_loader.dataset
        with open(counts_path, 'w', encoding='utf-8') as f:
            f.write(f"Epoch {epoch} class sample counts (original vs augmented duplicates)\n")
            f.write(f"{'Class':<40}{'Original':>10}{'Aug':>10}{'Total':>10}\n")
            f.write("-"*72 + "\n")
            for i, cname in enumerate(classes):
                orig = ds.original_counts.get(i, 0)
                aug = ds.augmented_counts.get(i, 0)
                f.write(f"{cname:<40}{orig:>10}{aug:>10}{(orig+aug):>10}\n")
            total_orig = sum(ds.original_counts.values())
            total_aug = sum(ds.augmented_counts.values())
            f.write("-"*72 + "\n")
            f.write(f"{'TOTAL':<40}{total_orig:>10}{total_aug:>10}{(total_orig+total_aug):>10}\n")
        # also latest symlink-ish copy
        latest_path = os.path.join(save_dir, 'class_counts_latest.txt')
        try:
            import shutil
            shutil.copyfile(counts_path, latest_path)
        except Exception:
            pass

    epoch_times: List[float] = []
    total_batches_all_epochs = len(train_loader) * cfg.num_epochs
    global_batch_counter = 0

    for epoch in range(cfg.num_epochs):
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        total_batches = len(train_loader)
        epoch_start = time.time()

        for batch_idx, (clips, labels) in enumerate(train_loader):
            clips, labels = clips.to(device), labels.to(device)
            optimizer.zero_grad()
            outputs = model(clips)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()

            running_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()

            # Progress / ETA (epoch + global)
            global_batch_counter += 1
            pct_epoch = 100.0 * (batch_idx + 1) / total_batches
            elapsed_epoch = time.time() - epoch_start
            avg_batch_time_epoch = elapsed_epoch / (batch_idx + 1)
            remaining_batches_epoch = total_batches - (batch_idx + 1)
            eta_epoch = avg_batch_time_epoch * remaining_batches_epoch

            pct_global = 100.0 * global_batch_counter / max(1, total_batches_all_epochs)
            overall_elapsed = sum(epoch_times) + elapsed_epoch
            avg_batch_global = overall_elapsed / global_batch_counter
            remaining_global_batches = total_batches_all_epochs - global_batch_counter
            eta_global = avg_batch_global * remaining_global_batches

            print(
                f"Epoch {epoch+1}/{cfg.num_epochs} "
                f"[{batch_idx+1}/{total_batches} {pct_epoch:5.1f}% | Global {pct_global:5.1f}%] "
                f"loss={loss.item():.4f} ETA_epoch={eta_epoch:6.1f}s ETA_total={eta_global:7.1f}s",
                end='\r'
            )
        print()

        train_loss = running_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)
        epoch_dur = time.time() - epoch_start
        epoch_times.append(epoch_dur)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining_epochs = cfg.num_epochs - (epoch + 1)
        eta_total = remaining_epochs * avg_epoch
        print(
            f"Epoch [{epoch+1}/{cfg.num_epochs}] "
            f"loss={train_loss:.4f} acc={train_acc:.2f}% "
            f"| epoch_time={format_secs(epoch_dur)} | ETA_total={format_secs(eta_total)}"
        )

        # ---- NEW: save class counts this epoch ----
        save_class_counts(epoch + 1)

        # Periodic test
        model_was_best = False
        if (epoch+1) % 5 == 0 or (epoch+1)==cfg.num_epochs:
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
                model_was_best = True

        # Always save epoch checkpoint
        epoch_ckpt = os.path.join(save_dir, f'epoch_{epoch+1}.pth')
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'best_test_acc': best_acc,
            'classes': classes,
            'config': cfg.__dict__,
            'was_best': model_was_best,
        }, epoch_ckpt)

        # Update rolling last model
        torch.save({
            'model_state': model.state_dict(),
            'epoch': epoch+1,
            'train_loss': train_loss,
            'train_acc': train_acc,
            'best_test_acc': best_acc,
            'classes': classes,
            'config': cfg.__dict__,
        }, last_model_path)

    print("Training finished.")
    if os.path.isfile(best_model_path):
        print(f"Best model saved at: {best_model_path} (acc={best_acc:.2f}%)")
    print(f"Last model snapshot saved at: {last_model_path}")

if __name__ == '__main__':
    train_mobilenet_lstm()
