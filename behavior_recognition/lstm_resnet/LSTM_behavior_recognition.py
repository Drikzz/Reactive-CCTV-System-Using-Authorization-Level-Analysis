"""LSTM-based dataset + dataloader setup for behavior recognition from MP4 clips.

Enhanced version that combines clean structure with advanced features.
Supports proper train/test splits, caching, augmentation, and hyperparameter tuning.
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

# Try to import optuna for hyperparameter optimization
try:
    import optuna
except ImportError:
    optuna = None

# -------------------------- Configuration -------------------------- #
@dataclass
class LSTMConfig:
    # Basic paths
    root_dir: str = os.path.join('datasets', 'behavior_clips')
    cache_root: str = os.path.join('datasets', 'cache', 'LSTM') 
    
    # Video processing
    sequence_length: int = 32              # number of frames sampled per video
    img_size: int = 224                   # resize to this size
    frame_sampling: str = 'uniform'        # 'uniform' | 'stride' | 'random'
    
    # Dataset settings
    train_split: float = 0.8              # train/test split per class
    val_split: float = 0.2                # validation split from training data
    random_seed: int = 42
    batch_size: int = 16
    num_workers: int = 2                  # parallel loading workers
    max_videos_per_class: Optional[int] = 200  # cap per class
    
    # Model architecture
    hidden_size: int = 256                # LSTM hidden size
    num_layers: int = 2                   # LSTM layers
    bidirectional: bool = False
    dropout: float = 0.3                  # LSTM dropout (if num_layers>1)
    freeze_cnn: bool = True               # freeze pretrained CNN weights
    
    # Training parameters
    learning_rate: float = 1e-4
    num_epochs: int = 50
    early_stop_patience: int = 5
    
    # Caching and preprocessing
    enable_cache: bool = True
    pre_cache_before_training: bool = True  # preprocess all videos before training
    show_video_conversion_progress: bool = True
    cache_format: str = 'pt'              # 'pt' or 'npy'
    
    # Augmentation parameters
    aug_flip_p: float = 0.5
    aug_crop_scale: Tuple[float,float] = (0.8, 1.0)
    aug_crop_ratio: Tuple[float,float] = (0.9, 1.1)
    aug_brightness: Tuple[float,float] = (0.8, 1.2)
    aug_contrast: Tuple[float,float] = (0.8, 1.2)
    
    # Hyperparameter optimization
    use_optuna: bool = False
    optuna_trials: int = 10

# -------------------------- Helper Functions -------------------------- #

def format_secs(secs: float) -> str:
    """Format seconds into human-readable time string."""
    secs = float(secs)
    if secs < 60:
        return f"{secs:.1f}s"
    minutes, s = divmod(int(round(secs)), 60)
    if minutes < 60:
        return f"{minutes}m{s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m{s:02d}s"

def set_global_seed(seed: int):
    """Set random seeds for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)

def seed_worker(worker_id: int):
    """Ensure each dataloader worker has a deterministic but different seed."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def _project_root() -> str:
    """Get the absolute path to the project root."""
    return os.path.abspath(os.path.join(os.path.dirname(__file__), '..'))

def _abs_under_root(p: str) -> str:
    """Convert relative path to absolute path under project root."""
    return p if os.path.isabs(p) else os.path.abspath(os.path.join(_project_root(), p))

# -------------------------- Dataset -------------------------- #

class BehaviorVideoDataset(Dataset):
    def __init__(
        self,
        video_label_pairs: List[Tuple[str, int]],  # or (path, label, aug_flag)
        classes: List[str],
        sequence_length: int,
        transform: Optional[Callable] = None,
        img_size: int = 224,
        frame_sampling: str = 'uniform',
        show_progress: bool = True,
        cache_root: Optional[str] = None,
        enable_cache: bool = True,
        dataset_root: Optional[str] = None,
        # augmentation parameters
        aug_flip_p: float = 0.5,
        aug_crop_scale: Tuple[float,float] = (0.8, 1.0),
        aug_crop_ratio: Tuple[float,float] = (0.9, 1.1),
        aug_brightness: Tuple[float,float] = (0.8, 1.2),
        aug_contrast: Tuple[float,float] = (0.8, 1.2),
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
        if self.enable_cache and self.cache_root:
            os.makedirs(self.cache_root, exist_ok=True)
        # unreadable
        self.unreadable_videos: List[str] = []
        self._unreadable_set: Set[str] = set()
        self._unreadable_lock = Lock()
        # per-sample fields
        self.paths: List[str] = []
        self.labels_only: List[int] = []
        self.aug_flags: List[bool] = []
        
        # Track original vs augmented counts per class
        self.original_counts: Dict[int, int] = {i: 0 for i in range(len(classes))}
        self.augmented_counts: Dict[int, int] = {i: 0 for i in range(len(classes))}
        
        for item in video_label_pairs:
            if len(item) == 2:
                pth, lab = item  # type: ignore
                augf = False
            else:
                pth, lab, augf = item  # type: ignore
            self.paths.append(pth)
            self.labels_only.append(int(lab))
            self.aug_flags.append(bool(augf))
            
            # Count originals vs augmented
            if 0 <= int(lab) < len(classes):
                if bool(augf):
                    self.augmented_counts[int(lab)] += 1
                else:
                    self.original_counts[int(lab)] += 1
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
        """Sample frame indices based on sampling strategy."""
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
        # else, derive relative path from dataset_root (if possible) and build cache path
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
        """Normalize clip using ImageNet mean/std."""
        # clip in [0,1], shape (T,3,H,W)
        return (clip - self._mean) / self._std

    def _augment_clip_inplace(self, clip: torch.Tensor) -> torch.Tensor:
        """Apply consistent augmentation across all frames in clip."""
        # clip in [0,1], shape (T,3,H,W). Apply same params across all frames.
        Tlen, C, H, W = clip.shape
        # flip
        do_flip = random.random() < self.aug_flip_p
        # crop params from the first frame using torchvision helper
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

    def _split_cache_index(self, path: str) -> Tuple[str, Optional[int]]:
        """Support addressing clips in aggregate cache files."""
        if "::idx=" in path:
            base, idx_str = path.rsplit("::idx=", 1)
            try:
                return base, int(idx_str)
            except Exception:
                return base, None
        return path, None

    def __getitem__(self, idx):
        video_path = self.paths[idx]
        label = self.labels_only[idx]
        do_aug = self.aug_flags[idx]

        # resolve '::idx=' addressing for aggregate cache files
        base_path, clip_idx_in_file = self._split_cache_index(video_path)

        cache_hit = False
        clip_pre: Optional[torch.Tensor] = None  # pre-normalization clip in [0,1]
        unreadable = False

        # ---- Prefer cache first ----
        cache_path = ""
        if self.enable_cache and self.cache_root:
            cache_path, path_is_cache = self._cache_path_for(base_path)

            def _as_tensor(obj) -> Optional[torch.Tensor]:
                if isinstance(obj, torch.Tensor):
                    return obj.float()
                if isinstance(obj, np.ndarray):
                    return torch.from_numpy(obj).float()
                if isinstance(obj, (list, tuple)) and len(obj) > 0:
                    elems = []
                    for it in obj:
                        t = _as_tensor(it)
                        if t is None:
                            return None
                        elems.append(t)
                    try:
                        return torch.stack(elems, dim=0).float()
                    except Exception:
                        return None
                if isinstance(obj, dict):
                    for k in ("clip", "tensor", "data", "clips", "clip_list", "clips_tensor", "video"):
                        if k in obj:
                            return _as_tensor(obj[k])
                return None

            def _load_cache(path: str) -> Optional[torch.Tensor]:
                try:
                    if path.lower().endswith(".npy") and os.path.isfile(path):
                        arr = np.load(path, allow_pickle=False, mmap_mode='r')
                        return _as_tensor(np.array(arr, copy=False))
                    if path.lower().endswith(".pt") and os.path.isfile(path):
                        obj = torch.load(path, map_location='cpu')
                        return _as_tensor(obj)
                    return None
                except Exception:
                    return None

            # Try primary cache path
            clip_pre = _load_cache(cache_path)
            # If not found and original was not already a cache file, try alternate extension
            if clip_pre is None and not base_path.lower().endswith((".pt", ".npy")):
                alt = cache_path[:-4] + (".npy" if self.cache_ext == ".pt" else ".pt")
                clip_pre = _load_cache(alt)
                if clip_pre is not None:
                    cache_path = alt
            cache_hit = clip_pre is not None

        # ---- Select specific clip from aggregate tensors if needed ----
        if clip_pre is not None:
            if isinstance(clip_pre, torch.Tensor):
                if clip_pre.dim() == 5:
                    # (N,T,3,H,W): select index if provided, else deterministic
                    if clip_idx_in_file is not None:
                        if 0 <= clip_idx_in_file < clip_pre.size(0):
                            clip_pre = clip_pre[clip_idx_in_file]
                        else:
                            clip_pre = clip_pre[0]
                    else:
                        N = clip_pre.size(0)
                        sel = (abs(hash(base_path)) + idx) % max(1, N)
                        clip_pre = clip_pre[sel]
                elif clip_pre.dim() == 3:
                    # (3,H,W) -> tile to sequence
                    clip_pre = clip_pre.unsqueeze(0).repeat(self.sequence_length, 1, 1, 1)

        # ---- Fallback: decode from video and save cache if enabled ----
        if clip_pre is None:
            # If base path is a cache file but missing, mark unreadable
            if base_path.lower().endswith((".npy", ".pt")):
                unreadable = True
            else:
                cap = cv2.VideoCapture(base_path)
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
                        # save cache for future runs
                        if self.enable_cache and self.cache_root and cache_path:
                            try:
                                if self.cache_ext == ".pt":
                                    # note: saving only the selected sequence
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

        # ---- Detect if cache is already normalized (from old pipeline) ----
        is_pre_normalized = False
        try:
            # old cache saved tensors after Normalize => typically outside [0,1]
            _min, _max = float(clip_pre.min()), float(clip_pre.max())
            is_pre_normalized = (_min < -0.05) or (_max > 1.05)
        except Exception:
            is_pre_normalized = False

        # ---- Apply augment/normalize respecting pre-normalized caches ----
        if is_pre_normalized:
            # Use as-is to match old cache exactly; skip augmentation and normalization
            clip = clip_pre
            origin = 'CACHE(NORMED)' if cache_hit else ('DECODE(NORMED)' if not unreadable else 'UNREADABLE')
        else:
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
                    print(f"Video tensor ready {self._processed_count}/{self._total_videos} ({pct:5.1f}%) [{origin}] - {os.path.basename(base_path)}")

        return clip, label

# -------------------------- Video Discovery and Dataset Functions -------------------------- #

def discover_videos(root_dir: str, max_videos_per_class: Optional[int] = None) -> Tuple[List[Tuple[str, int]], List[str]]:
    """Discover videos from a directory structure with categories and classes."""
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

def discover_cached_clips(cache_root: str, max_videos_per_class: Optional[int] = None) -> Tuple[List[Tuple[str,int]], List[str]]:
    """Discover cached clips from cache directory structure."""
    if not os.path.isdir(cache_root):
        raise FileNotFoundError(f"Cache directory not found: {cache_root}")

    def count_clips_in_cache(path: str) -> int:
        try:
            if path.lower().endswith(".npy"):
                arr = np.load(path, allow_pickle=False, mmap_mode='r')
                if isinstance(arr, np.memmap):
                    arr = np.asarray(arr)
                if isinstance(arr, np.ndarray) and arr.ndim >= 4:
                    # (N,T,3,H,W) or (T,3,H,W)
                    return int(arr.shape[0]) if arr.ndim == 5 else 1
                return 1
            if path.lower().endswith(".pt"):
                obj = torch.load(path, map_location='cpu')
                if isinstance(obj, torch.Tensor):
                    if obj.dim() == 5:
                        return int(obj.size(0))
                    return 1
                if isinstance(obj, (list, tuple)):
                    # list of clips
                    return len(obj)
                if isinstance(obj, dict):
                    for k in ("clips", "clip_list", "data", "tensor", "clip"):
                        if k in obj:
                            v = obj[k]
                            if isinstance(v, torch.Tensor) and v.dim() == 5:
                                return int(v.size(0))
                            if isinstance(v, (list, tuple)):
                                return len(v)
                            return 1
                return 1
        except Exception:
            return 1
        return 1

    records: List[Tuple[str, str]] = []  # (path, class_key)
    for dirpath, _dirnames, filenames in os.walk(cache_root):
        files = [f for f in filenames if f.lower().endswith((".pt", ".npy"))]
        if not files:
            continue
        rel_dir = os.path.relpath(dirpath, cache_root).replace("\\", "/")
        parts = [] if rel_dir in ("", ".", os.curdir) else rel_dir.split("/")

        for f in sorted(files):
            path = os.path.join(dirpath, f)
            stem, _ext = os.path.splitext(f)
            cls_key: Optional[str] = None
            # categories/<category>/<class>.pt -> "category/class"
            if len(parts) >= 2 and parts[0].lower() in ("categories", "category"):
                category = parts[1]
                class_name = stem
                cls_key = f"{category}/{class_name}"
            # fallback: category/class/<file>.pt -> "category/class"
            elif len(parts) >= 2:
                cls_key = f"{parts[0]}/{parts[1]}"
            # fallback: category/<file>.pt -> "category/<file-stem>"
            elif len(parts) == 1 and parts[0] not in (".", ""):
                cls_key = f"{parts[0]}/{stem}"
            else:
                cls_key = stem  # default

            records.append((path, cls_key))

    if not records:
        raise RuntimeError(f"No cached clips (.pt/.npy) found under {cache_root}")

    # group by class and expand per-clip indices
    by_class: Dict[str, List[str]] = {}
    for path, cls in records:
        by_class.setdefault(cls, []).append(path)

    classes = sorted(by_class.keys())
    cls_to_idx = {c: i for i, c in enumerate(classes)}

    video_label_pairs: List[Tuple[str, int]] = []
    for cls in classes:
        paths = sorted(by_class[cls])
        for p in paths:
            n = count_clips_in_cache(p)
            if max_videos_per_class is not None:
                # cap total clips contributed by this file
                n = min(n, max_videos_per_class)
            if n <= 1:
                video_label_pairs.append((p, cls_to_idx[cls]))
            else:
                # expand to address each clip individually
                for i in range(n):
                    video_label_pairs.append((f"{p}::idx={i}", cls_to_idx[cls]))

    return video_label_pairs, classes

def make_splits(video_label_pairs: List[Tuple[str, int]], classes: List[str], train_split: float, seed: int) -> Tuple[List[Tuple[str,int]], List[Tuple[str,int]]]:
    """Split video-label pairs into train and test sets."""
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
    """Split training pairs into training and validation sets."""
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

def cap_and_augment_to_target(
    train_pairs: List[Tuple[str,int]],
    classes: List[str],
    target_per_class: int,
    seed: int
) -> List[Tuple[str,int,bool]]:
    """Cap each class to target and flag samples for augmentation to reach target."""
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

# -------------------------- Model -------------------------- #
class CNNLSTM(nn.Module):
    def __init__(self, num_classes: int, hidden_size: int = 256, num_layers: int = 2, 
                 bidirectional: bool = False, dropout: float = 0.3, freeze_cnn: bool = True):
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

# -------------------------- Dataset and Dataloader Building -------------------------- #

def build_dataloaders(cfg: LSTMConfig):
    """Build training and test dataloaders."""
    # resolve absolute paths for discovery and dataset construction
    abs_root = _abs_under_root(cfg.root_dir)
    abs_cache = _abs_under_root(cfg.cache_root)

    # Prefer cache discovery first; fallback to videos
    try:
        video_label_pairs, classes = discover_cached_clips(abs_cache, cfg.max_videos_per_class)
    except Exception:
        video_label_pairs, classes = discover_videos(abs_root, cfg.max_videos_per_class)

    train_pairs, test_pairs = make_splits(video_label_pairs, classes, cfg.train_split, cfg.random_seed)
    # cap and oversample (training only) if requested
    if cfg.max_videos_per_class is not None:
        train_pairs_aug = cap_and_augment_to_target(train_pairs, classes, cfg.max_videos_per_class, cfg.random_seed)
    else:
        # keep exact counts; no augmentation flag
        train_pairs_aug = train_pairs

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

    # Use weighted sampling to balance classes
    class_counts = train_ds.class_counts
    sample_weights = [1.0 / max(1, class_counts[lab]) for lab in train_ds.labels_only]
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), 
                                   num_samples=len(train_ds), replacement=True)

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
    """Build training, validation and test dataloaders."""
    # resolve absolute paths
    abs_root = _abs_under_root(cfg.root_dir)
    abs_cache = _abs_under_root(cfg.cache_root)

    # Prefer cache discovery first; fallback to videos
    try:
        video_label_pairs, classes = discover_cached_clips(abs_cache, cfg.max_videos_per_class)
    except Exception:
        video_label_pairs, classes = discover_videos(abs_root, cfg.max_videos_per_class)

    train_pairs, test_pairs = make_splits(video_label_pairs, classes, cfg.train_split, cfg.random_seed)
    train_sub, val_sub = split_train_val_pairs(train_pairs, classes, cfg.val_split, cfg.random_seed)
    if cfg.max_videos_per_class is not None:
        train_pairs_aug = cap_and_augment_to_target(train_sub, classes, cfg.max_videos_per_class, cfg.random_seed)
    else:
        train_pairs_aug = train_sub

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
    sampler = WeightedRandomSampler(torch.as_tensor(sample_weights, dtype=torch.double), 
                                   num_samples=len(train_ds), replacement=True)

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
    """Run a single epoch of training or evaluation."""
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

# -------------------------- Optuna Hyperparameter Tuning -------------------------- #

def tune_with_optuna(cfg: LSTMConfig):
    """Use Optuna to find optimal hyperparameters."""
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

# -------------------------- Main Training Function -------------------------- #

def train_model():
    """Main training function."""
    # Initialize configuration
    cfg = LSTMConfig()
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seed(cfg.random_seed)
    print(f"Using device: {device}")
    
    # If hyperparameter tuning is requested, run Optuna
    if cfg.use_optuna:
        tune_with_optuna(cfg)
        return

    # Build dataloaders
    try:
        train_loader, test_loader, classes = build_dataloaders(cfg)
    except Exception as e:
        print(f"Dataset build failed: {e}\nPopulate datasets/behavior_clips/ or ensure cached .pt/.npy exists in {cfg.cache_root}.")
        return
    
    num_classes = len(classes)
    print(f"Discovered classes: {classes} (n={num_classes})")

    # Initialize model
    model = CNNLSTM(
        num_classes=num_classes,
        hidden_size=cfg.hidden_size,
        num_layers=cfg.num_layers,
        bidirectional=cfg.bidirectional,
        dropout=cfg.dropout,
        freeze_cnn=cfg.freeze_cnn
    ).to(device)

    # Setup loss and optimizer with class balancing
    class_counts = getattr(train_loader.dataset, "class_counts", [0]*num_classes)
    class_weights = torch.tensor(
        [1.0 / max(1, c) for c in class_counts],
        dtype=torch.float, device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(filter(lambda p: p.requires_grad, model.parameters()), lr=cfg.learning_rate)

    # Prepare model save directory
    model_name = os.path.splitext(os.path.basename(__file__))[0]
    save_dir = os.path.join('models', model_name)
    os.makedirs(save_dir, exist_ok=True)

    # Save configuration
    config_path = os.path.join(save_dir, 'config.json')
    if not os.path.isfile(config_path):
        with open(config_path, 'w', encoding='utf-8') as f:
            json.dump(cfg.__dict__, f, indent=2)

    # Save classes mapping
    classes_path = os.path.join(save_dir, 'classes.json')
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump({i: c for i, c in enumerate(classes)}, f, indent=2)

    # Pre-cache all videos if requested
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
        # Report unreadable videos
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

    # Training loop
    best_acc = -1.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')
    epoch_times = []
    
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        
        # Training phase
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
            print(f"Epoch {epoch+1}/{cfg.num_epochs} | Batch {batch_idx+1}/{total_batches} ({pct:5.1f}%) - loss={loss.item():.4f}", end='\r')
        
        # Print epoch summary
        print()  # new line after progress
        train_loss = running_loss / max(1, total)
        train_acc = 100.0 * correct / max(1, total)
        epoch_dur = time.time() - epoch_start
        epoch_times.append(epoch_dur)
        avg_epoch = sum(epoch_times) / len(epoch_times)
        remaining = cfg.num_epochs - (epoch + 1)
        eta_total = remaining * avg_epoch
        print(f"Epoch [{epoch+1}/{cfg.num_epochs}] loss={train_loss:.4f} acc={train_acc:.2f}% | time={format_secs(epoch_dur)} | ETA={format_secs(eta_total)}")
        
        # Evaluation phase (every 5 epochs or last epoch)
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
            
            test_acc = 100.0 * correct_eval / max(1, total_eval)
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

    print("Training finished.")
    if os.path.isfile(best_model_path):
        print(f"Best model saved at: {best_model_path} (acc={best_acc:.2f}%)")
    print(f"Last model snapshot saved at: {last_model_path}")
    
    # Save class sample counts to a text file
    counts_path = os.path.join(save_dir, 'class_counts.txt')
    with open(counts_path, 'w', encoding='utf-8') as f:
        f.write(f"Class counts for training dataset:\n")
        f.write(f"{'Class':<30} {'Original':<10} {'Augmented':<10} {'Total':<10}\n")
        f.write("-" * 60 + "\n")
        
        for i, class_name in enumerate(classes):
            orig = train_loader.dataset.original_counts.get(i, 0)
            aug = train_loader.dataset.augmented_counts.get(i, 0)
            total = orig + aug
            f.write(f"{class_name:<30} {orig:<10} {aug:<10} {total:<10}\n")
        
        # Add totals row
        total_orig = sum(train_loader.dataset.original_counts.values())
        total_aug = sum(train_loader.dataset.augmented_counts.values())
        total_all = total_orig + total_aug
        f.write("-" * 60 + "\n")
        f.write(f"{'TOTAL':<30} {total_orig:<10} {total_aug:<10} {total_all:<10}\n")
    
    print(f"Class sample counts saved to: {counts_path}")

if __name__ == '__main__':
    train_model()
