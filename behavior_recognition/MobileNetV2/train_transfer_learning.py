"""
Training script: MobileNetV2 transfer learning for behavior recognition

Features:
- MobileNetV2 pretrained on ImageNet, fully fine-tuned
- Focused augmentation to reduce background bias
- Tracks per-class accuracy, saves best checkpoint by val accuracy
- Plots/saves training curves (png) and a short text summary
- Clean console logs suitable for inclusion in thesis

Suggested extras (optional): label smoothing, dropout, mixup. These are noted at the bottom.
"""

import os
import sys
from pathlib import Path
import argparse
import time
from collections import Counter

import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
from tqdm import tqdm

# Project root (now two levels up since we're in MobileNetV2 subfolder)
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR))


def create_focused_augmentation():
    """Focused augmentation for person crops."""
    return transforms.Compose([
        transforms.Resize((256, 256)),
        transforms.RandomResizedCrop(224, scale=(0.8, 1.0)),
        transforms.RandomHorizontalFlip(p=0.5),
        transforms.RandomRotation(degrees=15),
        transforms.ColorJitter(brightness=0.3, contrast=0.3, saturation=0.3, hue=0.1),
        transforms.RandomAffine(degrees=10, translate=(0.08, 0.08)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        transforms.RandomErasing(p=0.12, scale=(0.02, 0.08))
    ])


def create_val_transform():
    return transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])


def plot_and_save(history, save_dir, model_name):
    save_dir.mkdir(parents=True, exist_ok=True)
    epochs = range(1, len(history['train_loss']) + 1)

    plt.style.use('seaborn-v0_8-darkgrid')
    fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(14, 5))

    ax1.plot(epochs, history['train_loss'], 'b-o', label='Train Loss')
    ax1.plot(epochs, history['val_loss'], 'r-s', label='Val Loss')
    ax1.set_title('Loss')
    ax1.set_xlabel('Epoch')
    ax1.set_ylabel('Loss')
    ax1.legend()

    ax2.plot(epochs, history['train_acc'], 'b-o', label='Train Acc')
    ax2.plot(epochs, history['val_acc'], 'r-s', label='Val Acc')
    ax2.set_title('Accuracy')
    ax2.set_xlabel('Epoch')
    ax2.set_ylabel('Accuracy (%)')
    ax2.legend()

    fig.suptitle('Training Curves')
    plt.tight_layout()

    png = save_dir / model_name.replace('.pth', '_training_curves.png')
    plt.savefig(str(png), dpi=300, bbox_inches='tight')
    plt.close()

    # Write simple summary text
    summary_path = save_dir / model_name.replace('.pth', '_training_summary.txt')
    with open(summary_path, 'w') as f:
        f.write('='*60 + '\n')
        f.write('TRAINING SUMMARY\n')
        f.write('='*60 + '\n')
        f.write(f"Total Epochs: {len(epochs)}\n")
        best_val_acc = max(history['val_acc']) if history['val_acc'] else 0
        best_epoch = history['val_acc'].index(best_val_acc) + 1 if history['val_acc'] else 0
        f.write(f"Best Val Acc: {best_val_acc:.2f}% (Epoch {best_epoch})\n")
        f.write(f"Final Train Loss: {history['train_loss'][-1]:.4f}\n")
        f.write(f"Final Val Loss: {history['val_loss'][-1]:.4f}\n")
        f.write(f"Final Train Acc: {history['train_acc'][-1]:.2f}%\n")
        f.write(f"Final Val Acc: {history['val_acc'][-1]:.2f}%\n")
        gap = history['train_acc'][-1] - history['val_acc'][-1]
        f.write(f"Train-Val Gap: {gap:.2f}%\n")
        if gap < 5:
            f.write('Status: ✅ Excellent\n')
        elif gap < 10:
            f.write('Status: ✅ Good\n')
        elif gap < 15:
            f.write('Status: ⚠️ Moderate overfitting\n')
        else:
            f.write('Status: ❌ High overfitting\n')

    print(f"Saved plots: {png}")
    print(f"Saved summary: {summary_path}")


def train(args):
    device = args.device or ('cuda' if torch.cuda.is_available() else 'cpu')
    print('\n' + '='*60)
    print('TRANSFER LEARNING TRAINING')
    print('='*60)
    print(f'Device: {device}')
    print(f'Dataset: {args.data}')
    print(f'Epochs: {args.epochs} | Batch: {args.batch_size} | LR: {args.lr}')
    print('='*60 + '\n')

    data_root = Path(args.data)
    train_dir = data_root / 'train'
    val_dir = data_root / 'valid'
    test_dir = data_root / 'test'

    train_transform = create_focused_augmentation()
    val_transform = create_val_transform()

    train_ds = datasets.ImageFolder(root=train_dir, transform=train_transform)
    val_ds = datasets.ImageFolder(root=val_dir, transform=val_transform)
    test_ds = datasets.ImageFolder(root=test_dir, transform=val_transform)

    class_names = train_ds.classes
    num_classes = len(class_names)
    print(f"Classes: {class_names} | {num_classes} classes")
    print(f"Samples - train: {len(train_ds)}, val: {len(val_ds)}, test: {len(test_ds)}")

    train_loader = DataLoader(train_ds, batch_size=args.batch_size, shuffle=True, num_workers=4, pin_memory=True)
    val_loader = DataLoader(val_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)
    test_loader = DataLoader(test_ds, batch_size=args.batch_size, shuffle=False, num_workers=4, pin_memory=True)

    # Model
    model = models.mobilenet_v2(weights=models.MobileNet_V2_Weights.IMAGENET1K_V1)
    # Unfreeze all layers
    for p in model.parameters():
        p.requires_grad = True
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model = model.to(device)

    criterion = nn.CrossEntropyLoss()
    optimizer = optim.Adam(model.parameters(), lr=args.lr)
    scheduler = optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='max', factor=0.5, patience=5)

    history = {'train_loss': [], 'train_acc': [], 'val_loss': [], 'val_acc': []}
    best_val_acc = 0.0

    start_time = time.time()
    for epoch in range(args.epochs):
        print(f"Epoch {epoch+1}/{args.epochs}")
        print('-'*50)
        # Train
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
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.2f}%"})

        train_loss = running_loss / len(train_loader)
        train_acc = 100 * correct / total

        # Validate
        model.eval()
        running_loss = 0.0
        correct = 0
        total = 0
        class_correct = [0]*num_classes
        class_total = [0]*num_classes

        with torch.no_grad():
            pbar = tqdm(val_loader, desc='Validation')
            for inputs, labels in pbar:
                inputs, labels = inputs.to(device), labels.to(device)
                outputs = model(inputs)
                loss = criterion(outputs, labels)
                running_loss += loss.item()
                _, preds = torch.max(outputs, 1)
                total += labels.size(0)
                correct += (preds == labels).sum().item()
                for i in range(len(labels)):
                    label = labels[i].item()
                    class_total[label] += 1
                    if preds[i] == labels[i]:
                        class_correct[label] += 1
                pbar.set_postfix({'loss': f"{loss.item():.4f}", 'acc': f"{100*correct/total:.2f}%"})

        val_loss = running_loss / len(val_loader)
        val_acc = 100 * correct / total

        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)

        scheduler.step(val_acc)

        # Logging
        print(f"Epoch {epoch+1} Summary: Train Loss {train_loss:.4f} Train Acc {train_acc:.2f}% | Val Loss {val_loss:.4f} Val Acc {val_acc:.2f}%")
        print("Per-class validation accuracy:")
        for i, cname in enumerate(class_names):
            if class_total[i] > 0:
                acc_i = 100 * class_correct[i] / class_total[i]
                print(f"  {cname}: {acc_i:.1f}% ({class_correct[i]}/{class_total[i]})")

        # Save best
        if val_acc > best_val_acc:
            best_val_acc = val_acc
            save_dir = BASE_DIR / args.save_dir
            save_dir.mkdir(parents=True, exist_ok=True)
            ckpt_path = save_dir / args.model_name
            checkpoint = {
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'num_classes': num_classes,
                'class_names': class_names,
                'best_val_acc': best_val_acc,
                'history': history,
                'epoch': epoch+1,
                'model_type': 'mobilenet_v2'
            }
            torch.save(checkpoint, str(ckpt_path))
            print(f"  ✅ Saved best model: {ckpt_path} (Val Acc {best_val_acc:.2f}%)")

    # Final test evaluation
    print('\n' + '='*60)
    print('FINAL TEST EVALUATION')
    print('='*60)
    model.eval()
    correct = 0
    total = 0
    class_correct = [0]*num_classes
    class_total = [0]*num_classes

    with torch.no_grad():
        for inputs, labels in tqdm(test_loader, desc='Testing'):
            inputs, labels = inputs.to(device), labels.to(device)
            outputs = model(inputs)
            _, preds = torch.max(outputs, 1)
            total += labels.size(0)
            correct += (preds == labels).sum().item()
            for i in range(len(labels)):
                label = labels[i].item()
                class_total[label] += 1
                if preds[i] == labels[i]:
                    class_correct[label] += 1

    test_acc = 100 * correct / total
    print(f"Overall Test Accuracy: {test_acc:.2f}%")
    print('Per-class test accuracy:')
    for i, cname in enumerate(class_names):
        if class_total[i] > 0:
            acc_i = 100 * class_correct[i] / class_total[i]
            print(f"  {cname}: {acc_i:.1f}% ({class_correct[i]}/{class_total[i]})")

    # Save plots and summary
    save_dir = BASE_DIR / args.save_dir
    plot_and_save(history, save_dir, args.model_name)

    total_time = time.time() - start_time
    print(f"Total training time: {total_time/60:.2f} minutes")

    # Append final test evaluation to summary file
    summary_path = save_dir / args.model_name.replace('.pth', '_training_summary.txt')
    with open(summary_path, 'a') as f:
        f.write('\n' + '='*60 + '\n')
        f.write('FINAL TEST EVALUATION\n')
        f.write('='*60 + '\n')
        f.write(f"Overall Test Accuracy: {test_acc:.2f}%\n")
        f.write('Per-class test accuracy:\n')
        for i, cname in enumerate(class_names):
            if class_total[i] > 0:
                acc_i = 100 * class_correct[i] / class_total[i]
                f.write(f"  {cname}: {acc_i:.1f}% ({class_correct[i]}/{class_total[i]})\n")
        f.write(f"Saved plots: {save_dir / args.model_name.replace('.pth', '_training_curves.png')}\n")
        f.write(f"Saved summary: {summary_path}\n")
        f.write(f"Total training time: {total_time/60:.2f} minutes\n")


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train MobileNetV2 (transfer learning)')
    parser.add_argument('--data', type=str, default='datasets', help='Dataset root directory')
    parser.add_argument('--epochs', type=int, default=30, help='Number of epochs')
    parser.add_argument('--batch-size', type=int, default=16, help='Batch size')
    parser.add_argument('--lr', type=float, default=1e-4, help='Learning rate')
    parser.add_argument('--save-dir', type=str, default='models/mobilenetv2', help='Directory to save model and plots')
    parser.add_argument('--model-name', type=str, default='mobilenet_transfer.pth', help='Model filename')
    parser.add_argument('--device', type=str, default=None, help='cuda or cpu')

    args = parser.parse_args()
    train(args)

# Optional techniques notes:
# - Label smoothing: can help a bit for small datasets (use CrossEntropy with label_smoothing)
# - Dropout: add dropout before classifier to reduce overfitting
# - Mixup: lightweight mixup augmentation sometimes helps generalization
# - Weighted sampling: if classes are imbalanced, use WeightedRandomSampler
