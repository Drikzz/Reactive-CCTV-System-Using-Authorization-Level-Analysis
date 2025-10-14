"""
Enhanced training script with class balancing and fine-tuning options.
Addresses common issues: class imbalance, frozen backbone, and label consistency.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader, WeightedRandomSampler
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import time
from collections import Counter

# Add parent directory to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


def get_balanced_sampler(dataset):
    """
    Create a WeightedRandomSampler to balance classes.
    
    Args:
        dataset: PyTorch dataset with targets attribute
        
    Returns:
        WeightedRandomSampler: Sampler for balanced batches
    """
    # Count samples per class
    class_counts = Counter(dataset.targets)
    num_samples = len(dataset.targets)
    
    # Calculate weights for each class (inverse frequency)
    class_weights = {cls: num_samples / count for cls, count in class_counts.items()}
    
    # Assign weight to each sample
    sample_weights = [class_weights[target] for target in dataset.targets]
    
    # Create sampler
    sampler = WeightedRandomSampler(
        weights=sample_weights,
        num_samples=len(sample_weights),
        replacement=True
    )
    
    print("\n📊 Class Balancing Info:")
    print(f"  Total samples: {num_samples}")
    for cls, count in class_counts.items():
        weight = class_weights[cls]
        print(f"  Class {cls}: {count} samples (weight: {weight:.3f})")
    
    return sampler


def load_datasets_with_balancing(data_dir, batch_size=32, num_workers=4, use_balancing=True):
    """
    Load datasets with optional class balancing.
    
    Args:
        data_dir (str): Path to dataset directory
        batch_size (int): Batch size
        num_workers (int): Number of data loading workers
        use_balancing (bool): Whether to use WeightedRandomSampler for training
        
    Returns:
        dict: Data loaders and class information
    """
    data_dir = Path(data_dir)
    
    # Transforms
    train_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=10),
        transforms.ColorJitter(brightness=0.2, contrast=0.2, saturation=0.2, hue=0.1),
        transforms.RandomAffine(degrees=0, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    val_test_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    datasets_dict = {
        'train': datasets.ImageFolder(root=data_dir / 'train', transform=train_transform),
        'valid': datasets.ImageFolder(root=data_dir / 'valid', transform=val_test_transform),
        'test': datasets.ImageFolder(root=data_dir / 'test', transform=val_test_transform)
    }
    
    class_names = datasets_dict['train'].classes
    
    # Create sampler for training if balancing is enabled
    train_sampler = None
    train_shuffle = True
    
    if use_balancing:
        train_sampler = get_balanced_sampler(datasets_dict['train'])
        train_shuffle = False  # Sampler handles shuffling
    
    # Create data loaders
    dataloaders = {
        'train': DataLoader(
            datasets_dict['train'],
            batch_size=batch_size,
            shuffle=train_shuffle,
            sampler=train_sampler,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'valid': DataLoader(
            datasets_dict['valid'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        ),
        'test': DataLoader(
            datasets_dict['test'],
            batch_size=batch_size,
            shuffle=False,
            num_workers=num_workers,
            pin_memory=torch.cuda.is_available()
        )
    }
    
    # Print dataset info
    print("\n" + "="*50)
    print("DATASET INFORMATION")
    print("="*50)
    print(f"Classes: {class_names}")
    print(f"Number of classes: {len(class_names)}")
    
    for split in ['train', 'valid', 'test']:
        dataset = datasets_dict[split]
        class_counts = Counter(dataset.targets)
        print(f"\n{split.upper()} SET: {len(dataset)} total samples")
        for idx, class_name in enumerate(class_names):
            count = class_counts.get(idx, 0)
            pct = (count / len(dataset)) * 100
            print(f"  {class_name}: {count} ({pct:.1f}%)")
    
    print("="*50 + "\n")
    
    return {
        'loaders': dataloaders,
        'class_names': class_names,
        'num_classes': len(class_names),
        'datasets': datasets_dict
    }


def create_model(num_classes, freeze_backbone=False, unfreeze_last_n=2):
    """
    Create MobileNetV2 model with flexible freezing options.
    
    Args:
        num_classes (int): Number of output classes
        freeze_backbone (bool): Whether to freeze the backbone
        unfreeze_last_n (int): Number of last feature blocks to keep trainable
        
    Returns:
        model: MobileNetV2 model
    """
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    if freeze_backbone:
        # Freeze all backbone layers
        for param in model.features.parameters():
            param.requires_grad = False
        
        # Unfreeze last N blocks if specified
        if unfreeze_last_n > 0:
            num_blocks = len(model.features)
            for i in range(num_blocks - unfreeze_last_n, num_blocks):
                for param in model.features[i].parameters():
                    param.requires_grad = True
            
            print(f"\n🔓 Unfroze last {unfreeze_last_n} feature blocks for fine-tuning")
    else:
        print("\n🔓 All backbone layers are trainable")
    
    # Replace classifier head
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    
    # Count trainable parameters
    total_params = sum(p.numel() for p in model.parameters())
    trainable_params = sum(p.numel() for p in model.parameters() if p.requires_grad)
    
    print(f"Total parameters: {total_params:,}")
    print(f"Trainable parameters: {trainable_params:,} ({trainable_params/total_params*100:.1f}%)")
    
    return model


def train_enhanced(data_dir, num_epochs=20, batch_size=32, learning_rate=0.001,
                  save_dir='models/mobilenet', model_name='mobilenet_best.pth',
                  use_balancing=True, freeze_backbone=False, unfreeze_last_n=2,
                  device=None):
    """
    Enhanced training with class balancing and fine-tuning options.
    
    Args:
        data_dir (str): Path to dataset directory
        num_epochs (int): Number of epochs
        batch_size (int): Batch size
        learning_rate (float): Learning rate
        save_dir (str): Directory to save model
        model_name (str): Model filename
        use_balancing (bool): Use WeightedRandomSampler for class balancing
        freeze_backbone (bool): Freeze backbone layers
        unfreeze_last_n (int): Number of last blocks to keep trainable if freezing
        device (str): Device to use
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    
    print("\n" + "="*50)
    print("ENHANCED MOBILENETV2 TRAINING")
    print("="*50)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {data_dir}")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print(f"Class Balancing: {'Enabled' if use_balancing else 'Disabled'}")
    print(f"Backbone: {'Partially Frozen' if freeze_backbone else 'Fully Trainable'}")
    if freeze_backbone and unfreeze_last_n > 0:
        print(f"Unfrozen Blocks: Last {unfreeze_last_n}")
    print("="*50)
    
    # Load datasets
    data_info = load_datasets_with_balancing(
        data_dir,
        batch_size=batch_size,
        use_balancing=use_balancing
    )
    
    dataloaders = data_info['loaders']
    class_names = data_info['class_names']
    num_classes = data_info['num_classes']
    
    # Create model
    model = create_model(
        num_classes=num_classes,
        freeze_backbone=freeze_backbone,
        unfreeze_last_n=unfreeze_last_n
    )
    model = model.to(device)
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    
    # Learning rate scheduler
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=3
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0.0
    
    # Training loop
    print("\nStarting training...\n")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 50)
        
        # Training phase
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(dataloaders['train'], desc='Training')
        for inputs, labels in pbar:
            inputs, labels = inputs.to(device), labels.to(device)
            
            optimizer.zero_grad()
            outputs = model(inputs)
            loss = criterion(outputs, labels)
            loss.backward()
            optimizer.step()
            
            running_loss += loss.item()
            _, predicted = torch.max(outputs.data, 1)
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            pbar.set_postfix({
                'loss': f'{loss.item():.4f}',
                'acc': f'{100 * correct / total:.2f}%'
            })
        
        train_loss = running_loss / len(dataloaders['train'])
        train_acc = 100 * correct / total
        
        # Validation phase
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        with torch.no_grad():
            pbar = tqdm(dataloaders['valid'], desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                pbar.set_postfix({
                    'loss': f'{loss.item():.4f}',
                    'acc': f'{100 * correct / total:.2f}%'
                })
        
        val_loss = running_loss / len(dataloaders['valid'])
        val_acc = 100 * correct / total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        # Learning rate scheduling
        scheduler.step(val_acc)
        
        # Print epoch summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best model
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = Path(save_dir) / model_name
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes,
                'class_names': class_names,
                'best_val_acc': best_val_acc,
                'history': history,
                'epoch': epoch + 1
            }
            torch.save(checkpoint, str(save_path))
            print(f"  ✓ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Training complete
    training_time = time.time() - start_time
    print("\n" + "="*50)
    print("TRAINING COMPLETE")
    print("="*50)
    print(f"Total Training Time: {training_time / 60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved at: {Path(save_dir) / model_name}")
    print("="*50 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train MobileNetV2 with enhancements')
    parser.add_argument('--data', type=str, default='datasets',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=20,
                       help='Number of training epochs')
    parser.add_argument('--batch-size', type=int, default=32,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.001,
                       help='Learning rate')
    parser.add_argument('--no-balancing', action='store_true',
                       help='Disable class balancing')
    parser.add_argument('--freeze-backbone', action='store_true',
                       help='Freeze backbone (except last N blocks)')
    parser.add_argument('--unfreeze-last', type=int, default=2,
                       help='Number of last blocks to keep trainable when freezing')
    parser.add_argument('--save-dir', type=str, default='models/mobilenet',
                       help='Directory to save model')
    parser.add_argument('--model-name', type=str, default='mobilenet_best.pth',
                       help='Model filename')
    
    args = parser.parse_args()
    
    # Resolve paths
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    
    train_enhanced(
        data_dir=str(data_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        model_name=args.model_name,
        use_balancing=not args.no_balancing,
        freeze_backbone=args.freeze_backbone,
        unfreeze_last_n=args.unfreeze_last
    )
