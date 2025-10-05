"""Frame-based behavior recognition using MobileNetV2.

This script implements pure image classification on individual frames (no sequences/LSTM).
Frames are loaded from datasets/behavior_frames/ with train/val/test splits.

Key features:
- MobileNetV2 backbone with custom classifier head
- Per-frame classification (no temporal modeling)
- Caching, augmentation, and class balancing
- WeightedRandomSampler for handling class imbalance
- Validation tracking with early stopping
- Learning rate scheduler
"""
from __future__ import annotations
import os
import sys
import random
import json
import time
from dataclasses import dataclass, asdict
from typing import List, Tuple, Optional, Dict
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader, WeightedRandomSampler
from PIL import Image
from torchvision.models import mobilenet_v2


# -------------------------- Configuration -------------------------- #
@dataclass
class FrameClassifierConfig:
    """Configuration for frame-based MobileNetV2 classifier."""
    
    # Dataset paths
    root_dir: str = os.path.join('datasets', 'behavior_frames')
    cache_root: str = os.path.join('datasets', 'cache', 'frame_classifier')
    
    # Image settings
    img_size: int = 224
    
    # Training splits (already split in folder structure)
    # We'll use the existing train/val/test folders
    
    # Random seed
    random_seed: int = 42
    
    # DataLoader settings
    batch_size: int = 32
    num_workers: int = 4
    
    # Model settings
    pretrained: bool = True
    freeze_backbone: bool = False  # Set to True to only train classifier
    dropout: float = 0.3
    
    # Training settings
    learning_rate: float = 1e-4
    num_epochs: int = 50
    early_stop_patience: int = 7
    
    # Caching
    enable_cache: bool = True
    cache_format: str = 'pt'  # 'pt' or 'npy'
    
    # Augmentation (for training only)
    aug_flip_p: float = 0.5
    aug_rotation: float = 15.0
    aug_crop_scale: Tuple[float, float] = (0.8, 1.0)
    aug_crop_ratio: Tuple[float, float] = (0.9, 1.1)
    aug_brightness: float = 0.2
    aug_contrast: float = 0.2
    aug_saturation: float = 0.2
    aug_hue: float = 0.1
    
    # Learning rate scheduler
    use_scheduler: bool = True
    scheduler_step_size: int = 10
    scheduler_gamma: float = 0.5


# -------------------------- Helper Functions -------------------------- #

def format_secs(secs: float) -> str:
    """Format seconds into human-readable string."""
    secs = float(secs)
    if secs < 60:
        return f"{secs:.1f}s"
    minutes, s = divmod(int(round(secs)), 60)
    if minutes < 60:
        return f"{minutes}m{s:02d}s"
    hours, m = divmod(minutes, 60)
    return f"{hours}h{m:02d}m{s:02d}s"


def set_global_seed(seed: int):
    """Set random seed for reproducibility."""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    # For deterministic behavior (may impact performance)
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False


def seed_worker(worker_id: int):
    """Seed worker for DataLoader."""
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)


def _project_root() -> str:
    """Get project root directory."""
    # Get the directory of this script file
    script_dir = os.path.dirname(os.path.abspath(__file__))
    # Go up two levels: lstm_mobilenet -> behavior_recognition -> project_root
    return os.path.abspath(os.path.join(script_dir, '..', '..'))


def _abs_under_root(p: str) -> str:
    """Convert relative path to absolute under project root."""
    if os.path.isabs(p):
        return p
    
    # First try relative to project root
    root_path = os.path.join(_project_root(), p)
    if os.path.exists(root_path):
        return os.path.abspath(root_path)
    
    # If not found, try relative to current working directory
    cwd_path = os.path.abspath(p)
    if os.path.exists(cwd_path):
        return cwd_path
    
    # Default to root path even if it doesn't exist yet
    return os.path.abspath(root_path)


# -------------------------- Dataset -------------------------- #

class FrameDataset(Dataset):
    """Dataset for loading individual frames with optional caching and augmentation."""
    
    def __init__(
        self,
        image_label_pairs: List[Tuple[str, int]],
        classes: List[str],
        img_size: int = 224,
        is_training: bool = False,
        enable_cache: bool = True,
        cache_root: Optional[str] = None,
        cache_format: str = 'pt',
        # Augmentation parameters
        aug_flip_p: float = 0.5,
        aug_rotation: float = 15.0,
        aug_crop_scale: Tuple[float, float] = (0.8, 1.0),
        aug_crop_ratio: Tuple[float, float] = (0.9, 1.1),
        aug_brightness: float = 0.2,
        aug_contrast: float = 0.2,
        aug_saturation: float = 0.2,
        aug_hue: float = 0.1,
    ):
        """
        Args:
            image_label_pairs: List of (image_path, label_idx) tuples
            classes: List of class names
            img_size: Target image size (square)
            is_training: Whether this is training set (enables augmentation)
            enable_cache: Whether to cache preprocessed images
            cache_root: Root directory for cache
            cache_format: 'pt' or 'npy'
            aug_*: Augmentation parameters (only used if is_training=True)
        """
        self.image_paths = [p for p, _ in image_label_pairs]
        self.labels = [l for _, l in image_label_pairs]
        self.classes = classes
        self.img_size = img_size
        self.is_training = is_training
        self.enable_cache = enable_cache and (cache_root is not None)
        self.cache_root = cache_root
        self.cache_ext = '.pt' if cache_format.lower() == 'pt' else '.npy'
        
        if self.enable_cache and self.cache_root:
            os.makedirs(self.cache_root, exist_ok=True)
        
        # Base preprocessing (resize + to tensor)
        self.base_transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
        ])
        
        # Normalization (ImageNet stats for MobileNet)
        self.normalize = T.Normalize(
            mean=[0.485, 0.456, 0.406],
            std=[0.229, 0.224, 0.225]
        )
        
        # Training augmentation
        if is_training:
            self.augmentation = T.Compose([
                T.RandomHorizontalFlip(p=aug_flip_p),
                T.RandomRotation(degrees=aug_rotation),
                T.RandomResizedCrop(
                    img_size,
                    scale=aug_crop_scale,
                    ratio=aug_crop_ratio
                ),
                T.ColorJitter(
                    brightness=aug_brightness,
                    contrast=aug_contrast,
                    saturation=aug_saturation,
                    hue=aug_hue
                ),
            ])
        else:
            self.augmentation = None
        
        # Count samples per class
        self.class_counts = [0] * len(classes)
        for label in self.labels:
            self.class_counts[label] += 1
    
    def _get_cache_path(self, image_path: str) -> str:
        """Generate cache file path for an image."""
        # Create a unique cache filename based on image path and settings
        rel_path = os.path.relpath(image_path, start=_project_root())
        rel_path_no_ext = os.path.splitext(rel_path)[0]
        cache_name = f"{os.path.basename(rel_path_no_ext)}_IMG{self.img_size}{self.cache_ext}"
        
        # Preserve directory structure in cache
        cache_dir = os.path.join(
            self.cache_root,
            os.path.dirname(rel_path_no_ext)
        )
        os.makedirs(cache_dir, exist_ok=True)
        return os.path.join(cache_dir, cache_name)
    
    def _load_and_preprocess(self, image_path: str) -> torch.Tensor:
        """Load and preprocess image (with caching if enabled)."""
        # Check cache first
        if self.enable_cache:
            cache_path = self._get_cache_path(image_path)
            if os.path.isfile(cache_path):
                try:
                    if self.cache_ext == '.pt':
                        tensor = torch.load(cache_path, weights_only=True)
                    else:
                        tensor = torch.from_numpy(np.load(cache_path))
                    return tensor
                except Exception:
                    pass  # If cache load fails, regenerate
        
        # Load and preprocess
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            # Return a blank image as fallback
            img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
        
        # Apply base transform (resize + to tensor)
        tensor = self.base_transform(img)
        
        # Save to cache (before augmentation and normalization)
        if self.enable_cache:
            cache_path = self._get_cache_path(image_path)
            try:
                if self.cache_ext == '.pt':
                    torch.save(tensor, cache_path)
                else:
                    np.save(cache_path, tensor.numpy())
            except Exception as e:
                print(f"Warning: Failed to cache {image_path}: {e}")
        
        return tensor
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int]:
        """Get a single frame and its label."""
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        # Load preprocessed tensor
        tensor = self._load_and_preprocess(image_path)
        
        # Apply augmentation if training
        if self.is_training and self.augmentation is not None:
            # Convert to PIL for augmentation
            pil_img = T.functional.to_pil_image(tensor)
            pil_img = self.augmentation(pil_img)
            tensor = T.functional.to_tensor(pil_img)
        
        # Normalize
        tensor = self.normalize(tensor)
        
        return tensor, label


# -------------------------- Model -------------------------- #

class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 backbone with custom classifier head for frame classification."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze backbone (only train classifier)
            dropout: Dropout probability in classifier
        """
        super().__init__()
        
        # Load MobileNetV2 backbone
        self.backbone = mobilenet_v2(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier head
        # MobileNetV2 feature dimension is 1280
        in_features = 1280
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)


# -------------------------- Data Loading -------------------------- #

def discover_frames(root_dir: str) -> Tuple[Dict[str, List[Tuple[str, int]]], List[str]]:
    """
    Discover frames from dataset directory structure.
    
    Expected structure:
        root_dir/
            train/
                class1/
                    img1.jpg
                    img2.jpg
                class2/
                    ...
            val/
                class1/
                    ...
            test/
                class1/
                    ...
    
    Returns:
        splits_dict: Dict with keys 'train', 'val', 'test' containing (image_path, label) lists
        classes: Sorted list of class names
    """
    if not os.path.isdir(root_dir):
        raise ValueError(f"Root directory not found: {root_dir}")
    
    splits = ['train', 'valid', 'test']
    splits_dict: Dict[str, List[Tuple[str, int]]] = {split: [] for split in splits}
    
    # Get classes from train folder
    train_dir = os.path.join(root_dir, 'train')
    if not os.path.isdir(train_dir):
        raise ValueError(f"Train directory not found: {train_dir}")
    
    classes = sorted([d for d in os.listdir(train_dir) 
                     if os.path.isdir(os.path.join(train_dir, d))])
    
    if not classes:
        raise ValueError(f"No class directories found in {train_dir}")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    # Discover frames for each split
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for split in splits:
        split_dir = os.path.join(root_dir, split)
        if not os.path.isdir(split_dir):
            print(f"Warning: {split} directory not found: {split_dir}")
            continue
        
        for class_name in classes:
            class_dir = os.path.join(split_dir, class_name)
            if not os.path.isdir(class_dir):
                continue
            
            label = class_to_idx[class_name]
            
            # Find all image files
            for filename in os.listdir(class_dir):
                ext = os.path.splitext(filename)[1].lower()
                if ext in valid_extensions:
                    image_path = os.path.join(class_dir, filename)
                    splits_dict[split].append((image_path, label))
    
    # Report statistics
    print(f"\nDiscovered {len(classes)} classes: {classes}")
    for split in splits:
        print(f"  {split}: {len(splits_dict[split])} frames")
    
    return splits_dict, classes


def build_dataloaders(cfg: FrameClassifierConfig):
    """Build train, val, and test dataloaders with class balancing."""
    abs_root = _abs_under_root(cfg.root_dir)
    abs_cache = _abs_under_root(cfg.cache_root)
    
    print(f"Project root: {_project_root()}")
    print(f"Dataset root: {abs_root}")
    print(f"Cache root: {abs_cache}")
    
    # Discover frames
    splits_dict, classes = discover_frames(abs_root)
    
    set_global_seed(cfg.random_seed)
    g = torch.Generator().manual_seed(cfg.random_seed)
    
    # Create datasets
    train_ds = FrameDataset(
        splits_dict['train'],
        classes,
        img_size=cfg.img_size,
        is_training=True,
        enable_cache=cfg.enable_cache,
        cache_root=abs_cache,
        cache_format=cfg.cache_format,
        aug_flip_p=cfg.aug_flip_p,
        aug_rotation=cfg.aug_rotation,
        aug_crop_scale=cfg.aug_crop_scale,
        aug_crop_ratio=cfg.aug_crop_ratio,
        aug_brightness=cfg.aug_brightness,
        aug_contrast=cfg.aug_contrast,
        aug_saturation=cfg.aug_saturation,
        aug_hue=cfg.aug_hue,
    )
    
    val_ds = FrameDataset(
        splits_dict['valid'],
        classes,
        img_size=cfg.img_size,
        is_training=False,
        enable_cache=cfg.enable_cache,
        cache_root=abs_cache,
        cache_format=cfg.cache_format,
    )
    
    test_ds = FrameDataset(
        splits_dict['test'],
        classes,
        img_size=cfg.img_size,
        is_training=False,
        enable_cache=cfg.enable_cache,
        cache_root=abs_cache,
        cache_format=cfg.cache_format,
    )
    
    # Print class distribution
    print("\nClass distribution:")
    for i, cls in enumerate(classes):
        train_count = train_ds.class_counts[i]
        val_count = val_ds.class_counts[i]
        test_count = test_ds.class_counts[i]
        print(f"  {cls}: train={train_count}, val={val_count}, test={test_count}")
    
    # Create WeightedRandomSampler for training (class balancing)
    class_counts = train_ds.class_counts
    sample_weights = [1.0 / max(1, class_counts[label]) for label in train_ds.labels]
    sampler = WeightedRandomSampler(
        torch.as_tensor(sample_weights, dtype=torch.double),
        num_samples=len(train_ds),
        replacement=True
    )
    
    # Create dataloaders
    train_loader = DataLoader(
        train_ds,
        batch_size=cfg.batch_size,
        sampler=sampler,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    val_loader = DataLoader(
        val_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    test_loader = DataLoader(
        test_ds,
        batch_size=cfg.batch_size,
        shuffle=False,
        num_workers=cfg.num_workers,
        pin_memory=True,
        worker_init_fn=seed_worker,
        generator=g
    )
    
    return train_loader, val_loader, test_loader, classes


# -------------------------- Training -------------------------- #

def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
    epoch: int
) -> Tuple[float, float]:
    """Train for one epoch."""
    model.train()
    running_loss = 0.0
    correct = 0
    total = 0
    
    for batch_idx, (inputs, labels) in enumerate(loader):
        inputs = inputs.to(device, non_blocking=True)
        labels = labels.to(device, non_blocking=True)
        
        optimizer.zero_grad()
        outputs = model(inputs)
        loss = criterion(outputs, labels)
        loss.backward()
        optimizer.step()
        
        running_loss += loss.item() * inputs.size(0)
        _, predicted = outputs.max(1)
        total += labels.size(0)
        correct += predicted.eq(labels).sum().item()
        
        if (batch_idx + 1) % 10 == 0:
            print(f"  Batch [{batch_idx + 1}/{len(loader)}], "
                  f"Loss: {loss.item():.4f}, "
                  f"Acc: {100. * correct / total:.2f}%")
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def validate(
    model: nn.Module,
    loader: DataLoader,
    criterion: nn.Module,
    device: torch.device
) -> Tuple[float, float]:
    """Validate model."""
    model.eval()
    running_loss = 0.0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for inputs, labels in loader:
            inputs = inputs.to(device, non_blocking=True)
            labels = labels.to(device, non_blocking=True)
            
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            
            running_loss += loss.item() * inputs.size(0)
            _, predicted = outputs.max(1)
            total += labels.size(0)
            correct += predicted.eq(labels).sum().item()
    
    # Handle empty loader
    if total == 0:
        return 0.0, 0.0
    
    avg_loss = running_loss / total
    accuracy = 100. * correct / total
    return avg_loss, accuracy


def train_frame_classifier():
    """Main training function."""
    cfg = FrameClassifierConfig()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    set_global_seed(cfg.random_seed)
    
    print("=" * 70)
    print("Frame-based Behavior Recognition with MobileNetV2")
    print("=" * 70)
    print(f"Device: {device}")
    print(f"Batch size: {cfg.batch_size}")
    print(f"Learning rate: {cfg.learning_rate}")
    print(f"Epochs: {cfg.num_epochs}")
    print(f"Early stopping patience: {cfg.early_stop_patience}")
    print(f"Image size: {cfg.img_size}")
    print(f"Pretrained: {cfg.pretrained}")
    print(f"Freeze backbone: {cfg.freeze_backbone}")
    print("=" * 70)
    
    # Build dataloaders
    print("\nBuilding dataloaders...")
    try:
        train_loader, val_loader, test_loader, classes = build_dataloaders(cfg)
    except Exception as e:
        print(f"Error building dataloaders: {e}")
        raise
    
    num_classes = len(classes)
    print(f"\nNumber of classes: {num_classes}")
    print(f"Classes: {classes}")
    
    # Build model
    print("\nBuilding model...")
    model = MobileNetV2Classifier(
        num_classes=num_classes,
        pretrained=cfg.pretrained,
        freeze_backbone=cfg.freeze_backbone,
        dropout=cfg.dropout
    ).to(device)
    
    # Count parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,}")
    
    # Loss and optimizer
    class_counts = train_loader.dataset.class_counts
    class_weights = torch.tensor(
        [1.0 / max(1, c) for c in class_counts],
        dtype=torch.float,
        device=device
    )
    criterion = nn.CrossEntropyLoss(weight=class_weights)
    optimizer = torch.optim.Adam(
        filter(lambda p: p.requires_grad, model.parameters()),
        lr=cfg.learning_rate
    )
    
    # Learning rate scheduler (optional)
    scheduler = None
    if cfg.use_scheduler:
        scheduler = torch.optim.lr_scheduler.StepLR(
            optimizer,
            step_size=cfg.scheduler_step_size,
            gamma=cfg.scheduler_gamma
        )
        print(f"Using StepLR scheduler: step_size={cfg.scheduler_step_size}, gamma={cfg.scheduler_gamma}")
    
    # Save directory
    save_dir = os.path.join('models', 'frame_classifier_mobilenet')
    os.makedirs(save_dir, exist_ok=True)
    
    # Save config and classes
    config_path = os.path.join(save_dir, 'config.json')
    with open(config_path, 'w', encoding='utf-8') as f:
        json.dump(asdict(cfg), f, indent=2)
    print(f"Config saved to: {config_path}")
    
    classes_path = os.path.join(save_dir, 'classes.json')
    with open(classes_path, 'w', encoding='utf-8') as f:
        json.dump(classes, f, indent=2)
    print(f"Classes saved to: {classes_path}")
    
    # Training loop
    best_val_acc = 0.0
    best_model_path = os.path.join(save_dir, 'best_model.pth')
    last_model_path = os.path.join(save_dir, 'last_model.pth')
    epochs_without_improvement = 0
    
    print("\n" + "=" * 70)
    print("Starting training...")
    print("=" * 70 + "\n")
    
    epoch_times = []
    
    for epoch in range(cfg.num_epochs):
        epoch_start = time.time()
        
        print(f"\nEpoch [{epoch + 1}/{cfg.num_epochs}]")
        print("-" * 70)
        
        # Train
        train_loss, train_acc = train_one_epoch(
            model, train_loader, criterion, optimizer, device, epoch
        )
        
        # Validate
        val_loss, val_acc = validate(model, val_loader, criterion, device)
        
        # Update scheduler
        if scheduler is not None:
            scheduler.step()
            current_lr = scheduler.get_last_lr()[0]
        else:
            current_lr = cfg.learning_rate
        
        epoch_time = time.time() - epoch_start
        epoch_times.append(epoch_time)
        avg_epoch_time = np.mean(epoch_times)
        remaining_epochs = cfg.num_epochs - (epoch + 1)
        eta = avg_epoch_time * remaining_epochs
        
        print(f"\n  Train Loss: {train_loss:.4f}, Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss: {val_loss:.4f}, Val Acc: {val_acc:.2f}%")
        print(f"  LR: {current_lr:.6f}")
        print(f"  Epoch time: {format_secs(epoch_time)}, ETA: {format_secs(eta)}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            torch.save(model.state_dict(), best_model_path)
            print(f"  ✓ New best validation accuracy: {best_val_acc:.2f}%")
            print(f"  ✓ Model saved to: {best_model_path}")
            epochs_without_improvement = 0
        else:
            epochs_without_improvement += 1
            print(f"  No improvement for {epochs_without_improvement} epoch(s)")
        
        # Save last model
        torch.save(model.state_dict(), last_model_path)
        
        # Early stopping
        if epochs_without_improvement >= cfg.early_stop_patience:
            print(f"\n⚠ Early stopping triggered after {epoch + 1} epochs")
            print(f"  No improvement for {cfg.early_stop_patience} consecutive epochs")
            break
    
    print("\n" + "=" * 70)
    print("Training completed!")
    print("=" * 70)
    print(f"Best validation accuracy: {best_val_acc:.2f}%")
    print(f"Best model saved to: {best_model_path}")
    print(f"Last model saved to: {last_model_path}")
    
    # Final evaluation on test set
    print("\n" + "=" * 70)
    print("Evaluating on test set...")
    print("=" * 70)
    
    # Load best model
    model.load_state_dict(torch.load(best_model_path, weights_only=True))
    test_loss, test_acc = validate(model, test_loader, criterion, device)
    
    print(f"Test Loss: {test_loss:.4f}")
    print(f"Test Accuracy: {test_acc:.2f}%")
    
    # Save test results
    results_path = os.path.join(save_dir, 'test_results.json')
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump({
            'test_loss': test_loss,
            'test_accuracy': test_acc,
            'best_val_accuracy': best_val_acc,
        }, f, indent=2)
    print(f"Test results saved to: {results_path}")


if __name__ == '__main__':
    train_frame_classifier()
