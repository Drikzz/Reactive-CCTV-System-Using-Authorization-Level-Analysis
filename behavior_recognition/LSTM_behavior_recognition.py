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
import gc

import cv2  # type: ignore
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
import torchvision.transforms as T
import torchvision.models as models
# Optuna for HPO
try:
    import optuna  # type: ignore
except Exception:
    optuna = None

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
    cache_root: str = os.path.join('datasets', 'cache', 'LSTM')  # root for cached tensors
    enable_cache: bool = True                            # toggle to disable caching
    pre_cache_before_training: bool = True               # run a full dataset pass to build cache and time it
    # ---- Optuna tuning toggles (keep defaults; only override LR, dropout, batch size) ----
    use_optuna: bool = False
    optuna_trials: int = 10
    val_split: float = 0.2
    early_stop_patience: int = 5
    # Balancing cap
    cap_per_class: int = 200
    # Augmentation probs/magnitude (training only, for augmented entries)
    aug_flip_p: float = 0.5
    aug_crop_scale: Tuple[float,float] = (0.8, 1.0)
    aug_crop_ratio: Tuple[float,float] = (0.9, 1.1)
    aug_brightness: Tuple[float,float] = (0.8, 1.2)
    aug_contrast: Tuple[float,float] = (0.8, 1.2)
    # New: cache format ('pt' or 'npy'), default to PyTorch
    cache_format: str = 'pt'

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
        video_label_pairs: List[Tuple[str, int]],  # or (path, label, aug_flag)
        classes: List[str],
        sequence_length: int,
        transform: Optional[Callable] = None,  # now expected to be normalization only
        img_size: int = 224,
        frame_sampling: str = 'uniform',
        show_progress: bool = True,
        cache_root: Optional[str] = None,
        enable_cache: bool = True,
        dataset_root: Optional[str] = None,
        # new optional augmentation knobs (used only when per-sample aug flag is True)
        aug_flip_p: float = 0.5,
        aug_crop_scale: Tuple[float,float] = (0.8, 1.0),
        aug_crop_ratio: Tuple[float,float] = (0.9, 1.1),
        aug_brightness: Tuple[float,float] = (0.8, 1.2),
        aug_contrast: Tuple[float,float] = (0.8, 1.2),
        # New:
        cache_format: str = 'pt',
    ):
        # core meta
        self.classes = classes
        self.sequence_length = sequence_length
        self.img_size = img_size
        # base preprocess: ToTensor+Resize only (cached), normalization applied later
        self._preprocess = T.Compose([
            T.ToTensor(),
            T.Resize((img_size, img_size)),
        ])
        # normalization params (use provided Normalize if passed)
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
        # progress tracking
        self._processed_videos = set()
        self._processed_count = 0
        self._progress_lock = Lock()
        self._total_videos = len(video_label_pairs)
        # caching
        self.enable_cache = enable_cache and (cache_root is not None)
        self.cache_root = cache_root
        self.dataset_root = dataset_root
        self.cache_ext = '.pt' if str(cache_format).lower() == 'pt' else '.npy'
        if self.enable_cache:
            os.makedirs(self.cache_root, exist_ok=True)
        # unreadable
        self.unreadable_videos: List[str] = []
        self._unreadable_set: Set[str] = set()
        self._unreadable_lock = Lock()
        # per-sample fields
        self.paths: List[str] = []
        self.labels_only: List[int] = []
        self.aug_flags: List[bool] = []
        for item in video_label_pairs:
            if len(item) == 2:
                pth, lab = item  # type: ignore
                augf = False
            else:
                pth, lab, augf = item  # type: ignore
            self.paths.append(pth)
            self.labels_only.append(int(lab))
            self.aug_flags.append(bool(augf))
        # class counts after capping/duplication
        n_classes = len(classes)
        self.class_counts: List[int] = [0] * n_classes
        for lab in self.labels_only:
            if 0 <= lab < n_classes:
                self.class_counts[lab] += 1
        # aug params
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

    def _cache_path_for(self, video_path: str) -> Tuple[str, bool]:
        """Return (cache_path, path_is_already_cache)."""
        if self.cache_root is None:
            return "", False
        lp = video_path.lower()
        if lp.endswith(".npy") or lp.endswith(".pt"):
            return video_path, True
        # else, derive relative path from dataset_root (if possible) and build .npy path
        if self.dataset_root and os.path.commonpath([os.path.abspath(video_path), os.path.abspath(self.dataset_root)]) == os.path.abspath(self.dataset_root):
            rel = os.path.relpath(video_path, self.dataset_root)
        else:
            rel = os.path.basename(video_path)
        rel_no_ext, _ = os.path.splitext(rel)
        rel_no_ext = rel_no_ext.replace('\\', '/').replace('..', '')
        rel_no_ext_base = os.path.basename(rel_no_ext)
        cache_filename = f"{rel_no_ext_base}_T{self.sequence_length}_S{self.frame_sampling}_IMG{self.img_size}{self.cache_ext}"
        subdir = os.path.dirname(rel_no_ext)  # keep class/category structure
        cache_dir = os.path.join(self.cache_root, subdir)
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, cache_filename), False

    def _normalize_clip(self, clip: torch.Tensor) -> torch.Tensor:
        # clip in [0,1], shape (T,3,H,W)
        return (clip - self._mean) / self._std

    def _augment_clip_inplace(self, clip: torch.Tensor) -> torch.Tensor:
        # clip in [0,1], shape (T,3,H,W). Apply same params across all frames.
        Tlen, C, H, W = clip.shape
        # flip
        do_flip = random.random() < self.aug_flip_p
        # crop params from the first frame using torchvision helper
        # convert one frame to PIL only to sample crop params
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
                fr = torch.flip(fr, dims=[2])  # horizontal
            fr = T.functional.resized_crop(
                fr, i, j, h, w, (self.img_size, self.img_size),
                interpolation=T.InterpolationMode.BILINEAR
            )
            fr = T.functional.adjust_brightness(fr, b_factor)
            fr = T.functional.adjust_contrast(fr, c_factor)
            out_frames.append(fr)
        return torch.stack(out_frames, dim=0)

    def __getitem__(self, idx):
        video_path = self.paths[idx]
        label = self.labels_only[idx]
        do_aug = self.aug_flags[idx]

        cache_hit = False
        clip_pre: Optional[torch.Tensor] = None  # pre-normalization clip in [0,1]
        unreadable = False

        # ---- Prefer .npy cache first ----
        cache_path = ""
        if self.enable_cache and self.cache_root:
            cache_path, path_is_cache = self._cache_path_for(video_path)

            def _load_cache(path: str) -> Optional[torch.Tensor]:
                try:
                    if path.lower().endswith(".npy") and os.path.isfile(path):
                        # mmap to reduce peak RAM during cache loads
                        arr = np.load(path, allow_pickle=False, mmap_mode='r')
                        return torch.from_numpy(np.array(arr, copy=False)).float()
                    if path.lower().endswith(".pt") and os.path.isfile(path):
                        obj = torch.load(path, map_location='cpu')
                        if isinstance(obj, torch.Tensor):
                            return obj.float()
                        if isinstance(obj, dict):
                            for k in ("clip", "tensor", "data"):
                                if k in obj:
                                    v = obj[k]
                                    if isinstance(v, np.ndarray):
                                        return torch.from_numpy(v).float()
                                    if isinstance(v, torch.Tensor):
                                        return v.float()
                        if isinstance(obj, np.ndarray):
                            return torch.from_numpy(obj).float()
                    return None
                except Exception:
                    return None

            # Try primary cache path
            clip_pre = _load_cache(cache_path)
            # If not found and path is not already cache (mp4 source), try alternate extension
            if clip_pre is None and not video_path.lower().endswith((".pt", ".npy")):
                alt = cache_path[:-4] + (".npy" if self.cache_ext == ".pt" else ".pt")
                clip_pre = _load_cache(alt)
                if clip_pre is not None:
                    cache_path = alt  # loaded from alt
            cache_hit = clip_pre is not None

        # ---- Fallback: decode from video and save cache if enabled ----
        if clip_pre is None:
            # If original path is already a cache path but missing, mark unreadable immediately
            if video_path.lower().endswith(".npy"):
                unreadable = True
            else:
                cap = cv2.VideoCapture(video_path)
                if not cap.isOpened():
                    unreadable = True
                else:
                    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
                    indices = self._sample_indices(frame_count)
                    wanted = set(indices)
                    cur = 0
                    grabbed_frames: Dict[int, torch.Tensor] = {}
                    while True:
                        ret, frame = cap.read()
                        if not ret:
                            break
                        if cur in wanted:
                            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                            pil_img = T.functional.to_pil_image(frame_rgb)
                            tensor = self._preprocess(pil_img)  # [0,1], no normalize
                            grabbed_frames[cur] = tensor
                            if len(grabbed_frames) == len(wanted):
                                break
                        cur += 1
                    cap.release()
                    if len(grabbed_frames) == 0:
                        unreadable = True
                    else:
                        ordered = [grabbed_frames.get(i) for i in indices]
                        if ordered[0] is None:
                            ordered[0] = torch.zeros(3, self.img_size, self.img_size)
                        for i in range(1, len(ordered)):
                            if ordered[i] is None:
                                ordered[i] = ordered[i - 1]
                        clip_pre = torch.stack(ordered, dim=0)  # (T,3,H,W)
                        # save npy cache for future runs
                        if self.enable_cache and self.cache_root and cache_path:
                            try:
                                if self.cache_ext == ".pt":
                                    torch.save(clip_pre.cpu(), cache_path)
                                else:
                                    np.save(cache_path, clip_pre.cpu().numpy())
                            except Exception as e:
                                if self.show_progress:
                                    print(f"Warning: failed to cache {cache_path}: {e}")

        # ---- If still missing, fallback to zeros ----
        if clip_pre is None:
            clip_pre = torch.zeros(self.sequence_length, 3, self.img_size, self.img_size)
            unreadable = True
            with self._unreadable_lock:
                if video_path not in self._unreadable_set:
                    self._unreadable_set.add(video_path)
                    self.unreadable_videos.append(video_path)

        # ---- Apply training-only augmentations on cached tensors, then normalize ----
        if do_aug:
            clip_aug = self._augment_clip_inplace(clip_pre)
            clip = self._normalize_clip(clip_aug)
            origin = 'CACHE+AUG' if cache_hit else ('DECODE+AUG' if not unreadable else 'UNREADABLE')
        else:
            clip = self._normalize_clip(clip_pre)
            origin = 'CACHE' if cache_hit else ('DECODE' if not unreadable else 'UNREADABLE')

        # progress (first-time)
        if self.show_progress:
            with self._progress_lock:
                if video_path not in self._processed_videos:
                    self._processed_videos.add(video_path)
                    self._processed_count += 1
                    pct = 100.0 * self._processed_count / max(1, self._total_videos)
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
    # Discover categories (e.g., neutral-actions, suspicious-actions)
    for category in sorted(os.listdir(root_dir)):
        category_path = os.path.join(root_dir, category)
        if os.path.isdir(category_path):
            # Discover classes within each category
            for class_name in sorted(os.listdir(category_path)):
                class_path = os.path.join(category_path, class_name)
                if os.path.isdir(class_path):
                    # Use 'category/class' as class label
                    full_class_name = f"{category}/{class_name}"
                    classes.append(full_class_name)
                    vids = [os.path.join(class_path, f) for f in os.listdir(class_path) if f.lower().endswith('.mp4')]
                    vids = sorted(vids)
                    if max_videos_per_class is not None:
                        vids = vids[:max_videos_per_class]
                    label = len(classes) - 1
                    for v in vids:
                        video_label_pairs.append((v, label))
    if not classes:
        raise RuntimeError(f"No class subdirectories with videos found under {root_dir}. Create structure root_dir/category/class/*.mp4")
    return video_label_pairs, classes


# Fallback: discover from cache if MP4s are absent (supports cache-only runs)
def discover_cached_clips(cache_root: str, max_videos_per_class: Optional[int] = None) -> Tuple[List[Tuple[str,int]], List[str]]:
    """
    Discover cached clips supporting:
      - cache_root/category/*.{pt,npy}
      - cache_root/category/class/*.{pt,npy}
    """
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"Cache directory not found: {cache_root}")
    records: List[Tuple[str, List[str]]] = []
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        files = [f for f in filenames if f.lower().endswith((".pt", ".npy"))]
        if not files:
            continue
        rel_dir = os.path.relpath(dirpath, cache_root)
        rel_dir = "." if rel_dir in ("", os.curdir) else rel_dir
        parts = rel_dir.replace("\\", "/").split("/")
        for f in sorted(files):
            records.append((os.path.join(dirpath, f), parts))
    if not records:
        raise RuntimeError(f"No cached clips (.pt/.npy) found under {cache_root}")
    has_two_levels = any(len(parts) >= 2 for _p, parts in records)
    def class_key(parts: List[str]) -> str:
        if has_two_levels and len(parts) >= 2:
            return f"{parts[0]}/{parts[1]}"
        elif len(parts) >= 1 and parts[0] not in (".", ""):
            return parts[0]
        else:
            return "default"
    by_class: Dict[str, List[str]] = {}
    for path, parts in records:
        by_class.setdefault(class_key(parts), []).append(path)
    classes = sorted(by_class.keys())
    cls_to_idx = {c: i for i, c in enumerate(classes)}
    video_label_pairs: List[Tuple[str, int]] = []
    for cls in classes:
        files = sorted(by_class[cls])
        if max_videos_per_class is not None:
            files = files[:max_videos_per_class]
        label = cls_to_idx[cls]
        for p in files:
            video_label_pairs.append((p, label))
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

def split_train_val_pairs(
    train_pairs: List[Tuple[str,int]],
    classes: List[str],
    val_ratio: float,
    seed: int
) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]:
    """Per-class split of given train_pairs into (train_sub, val_sub)."""
    rng = random.Random(seed)
    by_class: Dict[int, List[str]] = {i: [] for i in range(len(classes))}
    for path, label in train_pairs:
        by_class[label].append(path)
    train_sub: List[Tuple[str,int]] = []
    val_sub: List[Tuple[str,int]] = []
    for label, paths in by_class.items():
        paths = paths[:]  # copy
        rng.shuffle(paths)
        k_val = max(1, int(math.floor(len(paths) * val_ratio))) if len(paths) > 1 else 0
        val_paths = paths[:k_val]
        tr_paths = paths[k_val:]
        train_sub.extend([(p, label) for p in tr_paths])
        val_sub.extend([(p, label) for p in val_paths])
    return train_sub, val_sub

# ---- New: cap to target and add augmented duplicates for minority classes ----
def cap_and_augment_to_target(
    train_pairs: List[Tuple[str,int]],
    classes: List[str],
    target_per_class: int,
    seed: int
) -> List[Tuple[str,int,bool]]:
    """Cap each class to target_per_class; for minority classes, duplicate entries flagged as augmented until reaching target."""
    rng = random.Random(seed)
    by_class: Dict[int, List[str]] = {i: [] for i in range(len(classes))}
    for path, label in train_pairs:
        by_class[label].append(path)
    out: List[Tuple[str,int,bool]] = []
    for label, paths in by_class.items():
        paths = paths[:]
        rng.shuffle(paths)
        if len(paths) == 0:
            continue
        if len(paths) > target_per_class:
            kept = paths[:target_per_class]
            out.extend([(p, label, False) for p in kept])
        else:
            kept = paths[:]
            out.extend([(p, label, False) for p in kept])
            need = target_per_class - len(kept)
            if need > 0:
                # round-robin duplicate with augmentation flag
                i = 0
                while i < need:
                    out.append((kept[i % len(kept)], label, True))
                    i += 1
    return out

def compute_class_counts_from_pairs(pairs: List[Tuple[str,int]], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for _, lab in pairs:
        if 0 <= lab < num_classes:
            counts[lab] += 1
    return counts

def compute_class_counts_from_aug_pairs(pairs: List[Tuple[str,int,bool]], num_classes: int) -> List[int]:
    counts = [0] * num_classes
    for _, lab, _aug in pairs:
        if 0 <= lab < num_classes:
            counts[lab] += 1
    return counts

def set_global_seed(seed: int):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int):
    # Ensure each worker has a different but deterministic seed
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

# New: resolve paths relative to project root (parent of this file)
def _project_root() -> str:
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _abs_under_root(p: str) -> str:
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_project_root(), p))

# -------------------------- Build Datasets & Loaders -------------------------- #

def build_dataloaders(cfg: LSTMConfig):
    # resolve absolute paths for discovery and dataset construction
    abs_root = _abs_under_root(cfg.root_dir)
    abs_cache = _abs_under_root(cfg.cache_root)

    # try regular discovery first, else fallback to cache-only discovery
    try:
        video_label_pairs, classes = discover_videos(abs_root, cfg.max_videos_per_class)
    except Exception:
        video_label_pairs, classes = discover_cached_clips(abs_cache, cfg.max_videos_per_class)

    train_pairs, test_pairs = make_splits(video_label_pairs, classes, cfg.train_split, cfg.random_seed)
    # cap and oversample (training only)
    train_pairs_aug = cap_and_augment_to_target(train_pairs, classes, cfg.cap_per_class, cfg.random_seed)

    normalize_only = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    set_global_seed(cfg.random_seed)
    g = torch.Generator().manual_seed(cfg.random_seed)

    train_ds = BehaviorVideoDataset(
        train_pairs_aug, classes, cfg.sequence_length, normalize_only, cfg.img_size,
        cfg.frame_sampling, show_progress=cfg.show_video_conversion_progress,
        cache_root=abs_cache, enable_cache=cfg.enable_cache, dataset_root=abs_root,
        aug_flip_p=cfg.aug_flip_p, aug_crop_scale=cfg.aug_crop_scale, aug_crop_ratio=cfg.aug_crop_ratio,
        aug_brightness=cfg.aug_brightness, aug_contrast=cfg.aug_contrast,
        cache_format=cfg.cache_format
    )
    test_ds = BehaviorVideoDataset(
        test_pairs, classes, cfg.sequence_length, normalize_only, cfg.img_size,
        cfg.frame_sampling, show_progress=False,
        cache_root=abs_cache, enable_cache=cfg.enable_cache, dataset_root=abs_root,
        cache_format=cfg.cache_format
    )

    class_counts = train_ds.class_counts
    sample_weights = [1.0 / max(1, class_counts[lab]) for lab in train_ds.labels_only]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_ds), replacement=True)

    train_loader = DataLoader(
        train_ds, batch_size=cfg.batch_size, sampler=sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=cfg.batch_size, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, test_loader, classes

def build_train_val_test_loaders(cfg: LSTMConfig, batch_size_override: Optional[int] = None):
    # resolve absolute paths
    abs_root = _abs_under_root(cfg.root_dir)
    abs_cache = _abs_under_root(cfg.cache_root)

    # try regular discovery first, else fallback to cache-only discovery
    try:
        video_label_pairs, classes = discover_videos(abs_root, cfg.max_videos_per_class)
    except Exception:
        video_label_pairs, classes = discover_cached_clips(abs_cache, cfg.max_videos_per_class)

    train_pairs, test_pairs = make_splits(video_label_pairs, classes, cfg.train_split, cfg.random_seed)
    train_sub, val_sub = split_train_val_pairs(train_pairs, classes, cfg.val_split, cfg.random_seed)
    train_pairs_aug = cap_and_augment_to_target(train_sub, classes, cfg.cap_per_class, cfg.random_seed)

    normalize_only = T.Normalize(mean=[0.485,0.456,0.406], std=[0.229,0.224,0.225])

    set_global_seed(cfg.random_seed)
    g = torch.Generator().manual_seed(cfg.random_seed)

    train_ds = BehaviorVideoDataset(
        train_pairs_aug, classes, cfg.sequence_length, normalize_only, cfg.img_size,
        cfg.frame_sampling, show_progress=cfg.show_video_conversion_progress,
        cache_root=abs_cache, enable_cache=cfg.enable_cache, dataset_root=abs_root,
        aug_flip_p=cfg.aug_flip_p, aug_crop_scale=cfg.aug_crop_scale, aug_crop_ratio=cfg.aug_crop_ratio,
        aug_brightness=cfg.aug_brightness, aug_contrast=cfg.aug_contrast,
        cache_format=cfg.cache_format
    )
    val_ds = BehaviorVideoDataset(
        val_sub, classes, cfg.sequence_length, normalize_only, cfg.img_size,
        cfg.frame_sampling, show_progress=False,
        cache_root=abs_cache, enable_cache=cfg.enable_cache, dataset_root=abs_root,
        cache_format=cfg.cache_format
    )
    test_ds = BehaviorVideoDataset(
        test_pairs, classes, cfg.sequence_length, normalize_only, cfg.img_size,
        cfg.frame_sampling, show_progress=False,
        cache_root=abs_cache, enable_cache=cfg.enable_cache, dataset_root=abs_root,
        cache_format=cfg.cache_format
    )

    class_counts = train_ds.class_counts
    sample_weights = [1.0 / max(1, class_counts[lab]) for lab in train_ds.labels_only]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), num_samples=len(train_ds), replacement=True)

    bs = batch_size_override if batch_size_override is not None else cfg.batch_size
    train_loader = DataLoader(
        train_ds, batch_size=bs, sampler=sampler, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    val_loader = DataLoader(
        val_ds, batch_size=bs, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    test_loader = DataLoader(
        test_ds, batch_size=bs, shuffle=False,
        num_workers=cfg.num_workers, pin_memory=True, worker_init_fn=seed_worker, generator=g
    )
    return train_loader, val_loader, test_loader, classes

# -------------------------- Training / Evaluation Helpers -------------------------- #

def run_epoch(model: nn.Module, loader: DataLoader, criterion: nn.Module, optimizer: Optional[torch.optim.Optimizer], device: torch.device):
    """Train if optimizer is provided; otherwise evaluate. Returns (avg_loss, accuracy)."""
    if optimizer is None:
        model.eval()
        torch.set_grad_enabled(False)
    else:
        model.train()
        torch.set_grad_enabled(True)

    running_loss = 0.0
    correct = 0
    total = 0
    for clips, labels in loader:
        clips, labels = clips.to(device), labels.to(device)
        outputs = model(clips)
        loss = criterion(outputs, labels)
        if optimizer is not None:
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
        running_loss += loss.item() * labels.size(0)
        _, pred = outputs.max(1)
        total += labels.size(0)
        correct += pred.eq(labels).sum().item()

    avg_loss = running_loss / max(1, total)
    acc = 100.0 * correct / max(1, total)
    return avg_loss, acc

# -------------------------- Optuna Tuning -------------------------- #

def tune_with_optuna(cfg: LSTMConfig):
    if optuna is None:
        print("Optuna not installed. Run: pip install optuna")
        return

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Optuna tuning on device: {device}")

    # Prepare static datasets (recreated per batch size change to adjust loaders)
    def build_loaders_for_bs(bs: int):
        return build_train_val_test_loaders(cfg, batch_size_override=bs)

    def objective(trial: "optuna.trial.Trial") -> float:
        # Search space (override only LR, dropout, batch size)
        lr = trial.suggest_float("learning_rate", 1e-5, 1e-3, log=True)
        dropout = trial.suggest_float("dropout", 0.1, 0.7)
        bs_choices = sorted({cfg.batch_size, max(1, cfg.batch_size // 2), cfg.batch_size * 2})
        bs = trial.suggest_categorical("batch_size", bs_choices)

        train_loader, val_loader, _test_loader, classes = build_loaders_for_bs(bs)

        model = CNNLSTM(
            num_classes=len(classes),
            hidden_size=cfg.hidden_size,
            num_layers=cfg.num_layers,
            bidirectional=cfg.bidirectional,
            dropout=dropout,
            freeze_cnn=cfg.freeze_cnn
        ).to(device)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

        best_val_loss = float('inf')
        best_state = None
        patience = cfg.early_stop_patience
        epochs_no_improve = 0

        for epoch in range(cfg.num_epochs):
            _train_loss, _train_acc = run_epoch(model, train_loader, criterion, optimizer, device)
            val_loss, _val_acc = run_epoch(model, val_loader, criterion, None, device)

            # Early stopping on validation loss
            if val_loss + 1e-8 < best_val_loss:
                best_val_loss = val_loss
                best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
                epochs_no_improve = 0
            else:
                epochs_no_improve += 1
                if epochs_no_improve >= patience:
                    break

            # Report progress to Optuna
            trial.report(val_loss, epoch)
            if trial.should_prune():
                raise optuna.TrialPruned()

        # Restore best weights before returning objective
        if best_state is not None:
            model.load_state_dict(best_state)

        return best_val_loss

    study = optuna.create_study(direction="minimize")
    n_trials = min(cfg.optuna_trials, 10)  # hard cap at 10
    study.optimize(objective, n_trials=n_trials)

    print(f"Best trial: {study.best_trial.number}")
    best_params = study.best_trial.params
    print(f"Best params: {best_params}")

    # Evaluate validation accuracy with best params (retrain once with early stopping)
    bs = best_params.get("batch_size", cfg.batch_size)
    lr = best_params.get("learning_rate", cfg.learning_rate)
    dropout = best_params.get("dropout", cfg.dropout)

    train_loader, val_loader, _test_loader, classes = build_train_val_test_loaders(cfg, batch_size_override=bs)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model = CNNLSTM(
        num_classes=len(classes),
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        dropout=dropout,
        freeze_cnn=cfg.freeze_cnn
    ).to(device)
    criterion = nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=lr)

    best_state = None
    best_val_loss = float('inf')
    epochs_no_improve = 0
    for epoch in range(cfg.num_epochs):
        _train_loss, _ = run_epoch(model, train_loader, criterion, optimizer, device)
        val_loss, _ = run_epoch(model, val_loader, criterion, None, device)
        if val_loss + 1e-8 < best_val_loss:
            best_val_loss = val_loss
            best_state = {k: v.cpu().clone() for k, v in model.state_dict().items()}
            epochs_no_improve = 0
        else:
            epochs_no_improve += 1
            if epochs_no_improve >= cfg.early_stop_patience:
                break
    if best_state is not None:
        model.load_state_dict(best_state)
    _, val_acc = run_epoch(model, val_loader, criterion, None, device)
    print(f"Best validation accuracy: {val_acc:.2f}%")

# -------------------------- Example Training Loop -------------------------- #

def train_example():
    cfg = LSTMConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seed(cfg.random_seed)
    print(f"Using device: {device}")
    # If tuning requested, run Optuna and exit
    if cfg.use_optuna:
        tune_with_optuna(cfg)
        return
    try:
        train_loader, test_loader, classes = build_dataloaders(cfg)
    except Exception as e:
        print(f"Dataset build failed: {e}\nPopulate datasets/behavior_clips/ or ensure cached .pt/.npy exists in {cfg.cache_root}.")
        return
    num_classes = len(classes)
    print(f"Discovered classes: {classes} (n={num_classes})")

    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
        freeze_cnn=cfg.freeze_cnn
    ).to(device)

    # Class-balanced weighted loss: weights = 1.0 / class_counts (from training dataset)
    class_counts = getattr(train_loader.dataset, "class_counts", [0]*num_classes)
    class_weights = torch.tensor(
        [1.0 / max(1, c) for c in class_counts],
        dtype=torch.float, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)

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
    # Note: augmented training samples are applied on cached tensors each epoch.
    if cfg.pre_cache_before_training:
        print("Preprocessing: ensuring cache exists (one pass over datasets)...")
        t0 = time.time()
        try:
            # Stream through datasets without materializing lists
            for i in range(len(train_loader.dataset)):
                _ = train_loader.dataset[i]
            gc.collect()
            for i in range(len(test_loader.dataset)):
                _ = test_loader.dataset[i]
            gc.collect()
        except Exception as e:
            print(f"Warning during preprocessing pass: {e}")
        t1 = time.time()
        prep_secs = t1 - t0
        print(f"Preprocessing completed in {format_secs(prep_secs)} ({prep_secs:.1f}s)")
        # Report unreadable videos detected during preprocessing
        unreadable_all = list(dict.fromkeys(
            train_loader.dataset.unreadable_videos + test_loader.dataset.unreadable_videos
        ))
        if len(unreadable_all) > 0:
            print("Unreadable videos detected (replaced with black frames):")
            for p in unreadable_all:
                try:
                    rel = os.path.relpath(p, cfg.root_dir)
                except Exception:
                    rel = p
                print(f"  - {rel}")
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
            loss = criterion(outputs, labels)  # weighted loss
            loss.backward()
            optimizer.step()
            running_loss += loss.item() * labels.size(0)
            _, pred = outputs.max(1)
            total += labels.size(0)
            correct += pred.eq(labels).sum().item()
            # Progress percentage for current epoch
            pct = 100.0 * (batch_idx + 1) / total_batches
            print(f"Epoch {epoch+1}/{cfg.num_epochs} | Batch {batch_idx+1}/{total_batches} ({pct:5.1f}%) - batch_weighted_loss={loss.item():.4f}", end='\r')
        # After epoch, ensure newline
        print()
        train_loss = running_loss / max(1,total)
        train_acc = 100.0 * correct / max(1,total)
        epoch_dur = time.time() - epoch_start
        epoch_times.append(epoch_dur)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = cfg.num_epochs - (epoch + 1)
        eta_total = remaining * avg_epoch
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] weighted_loss={train_loss:.4f} acc={train_acc:.2f}% | time={format_secs(epoch_dur)} | ETA total={format_secs(eta_total)}")
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
