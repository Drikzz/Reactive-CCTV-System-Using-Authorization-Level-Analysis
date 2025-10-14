"""
Balanced training with FOCUSED augmentation to combat background bias.
Uses moderate augmentation that helps without destroying features.
"""

import os
import sys
from pathlib import Path
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import numpy as np
from tqdm import tqdm
import time
from collections import Counter

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


def create_balanced_augmentation():
    """
    Create BALANCED augmentation that helps with background bias
    without destroying the actual features we want to learn.
    """
    return transforms.Compose([
        transforms.Resize((256, 256)),
        # MODERATE crop - keeps most of the person but adds some variety
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        # Moderate rotation to change cabinet angles without distorting person
        transforms.RandomRotation(degrees=15),
        # Color jitter to make cabinet color less distinctive
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        # Small spatial changes
        transforms.RandomAffine(degrees=10, translate=(0.1, 0.1)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        # LIGHT random erasing - only small patches
        transforms.RandomErasing(p=0.15, scale=(0.02, 0.1))
    ])


def train_focused(data_dir, num_epochs=30, batch_size=16, learning_rate=0.0001,
                 save_dir='models/mobilenet', model_name='mobilenet_focused.pth',
                 device=None):
    """
    Train with focused, balanced augmentation.
    
    Args:
        data_dir (str): Path to dataset
        num_epochs (int): Training epochs
        batch_size (int): Batch size  
        learning_rate (float): Learning rate
        save_dir (str): Model save directory (relative to project root)
        model_name (str): Model filename
        device (str): Device
    """
    device = device or ('cuda' if torch.cuda.is_available() else 'cpu')
    data_dir = Path(data_dir)
    
    # Ensure save_dir is relative to project root (BASE_DIR)
    save_dir = BASE_DIR / save_dir if not Path(save_dir).is_absolute() else Path(save_dir)
    
    print("\n" + "="*70)
    print("FOCUSED AUGMENTATION TRAINING")
    print("="*70)
    print(f"Device: {torch.cuda.get_device_name(0) if torch.cuda.is_available() else 'CPU'}")
    print(f"Dataset: {data_dir}")
    print(f"Strategy: Balanced augmentation to reduce background bias")
    print(f"Epochs: {num_epochs}")
    print(f"Batch Size: {batch_size}")
    print(f"Learning Rate: {learning_rate}")
    print("="*70)
    
    # Create transforms
    train_transform = create_balanced_augmentation()
    
    val_transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    # Create datasets
    train_dataset = datasets.ImageFolder(
        root=data_dir / 'train',
        transform=train_transform
    )
    
    val_dataset = datasets.ImageFolder(
        root=data_dir / 'valid',
        transform=val_transform
    )
    
    test_dataset = datasets.ImageFolder(
        root=data_dir / 'test',
        transform=val_transform
    )
    
    class_names = train_dataset.classes
    num_classes = len(class_names)
    
    # Print dataset info
    print(f"\nClasses: {class_names}")
    print(f"Train samples: {len(train_dataset)}")
    print(f"Val samples: {len(val_dataset)}")
    print(f"Test samples: {len(test_dataset)}")
    
    # Data loaders
    train_loader = DataLoader(
        train_dataset,
        batch_size=batch_size,
        shuffle=True,
        num_workers=4,
        pin_memory=True
    )
    
    val_loader = DataLoader(
        val_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    test_loader = DataLoader(
        test_dataset,
        batch_size=batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Create model - FULLY TRAINABLE for better adaptation
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    
    # Make ALL layers trainable
    for param in model.parameters():
        param.requires_grad = True
    
    # Replace classifier
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)
    
    print(f"\n🔓 All model layers trainable for maximum adaptation")
    
    # Loss and optimizer
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=learning_rate)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='max', factor=0.5, patience=5
    )
    
    # Training history
    history = {
        'train_loss': [],
        'train_acc': [],
        'val_loss': [],
        'val_acc': []
    }
    best_val_acc = 0.0
    
    print("\n🚀 Starting focused training...\n")
    start_time = time.time()
    
    for epoch in range(num_epochs):
        print(f"\nEpoch {epoch + 1}/{num_epochs}")
        print("-" * 70)
        
        # Training
        model.train()
        running_loss = 0.0
        correct = 0
        total = 0
        
        pbar = tqdm(train_loader, desc='Training')
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
            
            pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total
        
        # Validation
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        
        # Per-class accuracy tracking
        class_correct = [0] * num_classes
        class_total = [0] * num_classes
        
        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                
                running_loss += loss.item()
                _, predicted = torch.max(outputs.data, 1)
                total += labels.size(0)
                correct += (predicted == labels).sum().item()
                
                # Per-class stats
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if predicted[i] == labels[i]:
                        class_correct[label] += 1
                
                pbar.set_postfix({'loss': f'{loss.item():.4f}', 'acc': f'{100*correct/total:.2f}%'})
        
        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total
        
        # Update history
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        scheduler.step(val_acc)
        
        # Print summary
        print(f"\nEpoch {epoch + 1} Summary:")
        print(f"  Train Loss: {train_loss:.4f} | Train Acc: {train_acc:.2f}%")
        print(f"  Val Loss:   {val_loss:.4f} | Val Acc:   {val_acc:.2f}%")
        print(f"  Per-Class Val Accuracy:")
        for i, class_name in enumerate(class_names):
            if class_total[i] > 0:
                class_acc = 100 * class_correct[i] / class_total[i]
                print(f"    {class_name}: {class_acc:.1f}% ({class_correct[i]}/{class_total[i]})")
        print(f"  LR: {optimizer.param_groups[0]['lr']:.6f}")
        
        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_path = save_dir / model_name
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
            print(f"  ✅ New best model saved! (Val Acc: {val_acc:.2f}%)")
    
    # Test evaluation
    print("\n" + "="*70)
    print("FINAL TEST EVALUATION")
    print("="*70)
    
    model.eval()
    correct = 0
    total = 0
    class_correct = [0] * num_classes
    class_total = [0] * num_classes
    
    with torch.no_grad():
        for inputs, labels in test_loader:
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, predicted = torch.max(outputs.data, 1)
            
            total += labels.size(0)
            correct += (predicted == labels).sum().item()
            
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if predicted[i] == labels[i]:
                    class_correct[label] += 1
    
    test_acc = 100 * correct / total
    print(f"\nOverall Test Accuracy: {test_acc:.2f}%")
    print(f"\nPer-Class Test Accuracy:")
    for i, class_name in enumerate(class_names):
        if class_total[i] > 0:
            class_acc = 100 * class_correct[i] / class_total[i]
            print(f"  {class_name}: {class_acc:.1f}% ({class_correct[i]}/{class_total[i]})")
    
    training_time = time.time() - start_time
    print(f"\nTotal Training Time: {training_time / 60:.2f} minutes")
    print(f"Best Validation Accuracy: {best_val_acc:.2f}%")
    print(f"Model saved: {Path(save_dir) / model_name}")
    print("="*70 + "\n")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Train with focused augmentation')
    parser.add_argument('--data', type=str, default='datasets',
                       help='Path to dataset directory')
    parser.add_argument('--epochs', type=int, default=30,
                       help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16,
                       help='Batch size')
    parser.add_argument('--lr', type=float, default=0.0001,
                       help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models/mobilenet',
                       help='Save directory')
    parser.add_argument('--model-name', type=str, default='mobilenet_focused.pth',
                       help='Model filename')
    
    args = parser.parse_args()
    
    data_dir = Path(args.data)
    if not data_dir.is_absolute():
        data_dir = BASE_DIR / data_dir
    
    train_focused(
        data_dir=str(data_dir),
        num_epochs=args.epochs,
        batch_size=args.batch_size,
        learning_rate=args.lr,
        save_dir=args.save_dir,
        model_name=args.model_name
    )
