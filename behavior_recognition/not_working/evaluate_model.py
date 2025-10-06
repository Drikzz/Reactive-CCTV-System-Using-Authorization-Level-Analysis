"""Diagnostic script to evaluate model performance and analyze predictions.

This script helps diagnose issues like:
- Class imbalance effects
- Confusion between classes
- Per-class accuracy
- Confidence distribution
"""
from __future__ import annotations
import os
import sys
import json
from pathlib import Path
from collections import Counter, defaultdict
from typing import List, Tuple, Dict

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from torch.utils.data import Dataset, DataLoader
from PIL import Image
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import confusion_matrix, classification_report
from tqdm import tqdm

# Add parent directory to path
sys.path.append(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from torchvision.models import mobilenet_v2


# -------------------------- Model Definition -------------------------- #

class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 backbone with custom classifier head for frame classification."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        super().__init__()
        self.backbone = mobilenet_v2(pretrained=pretrained)
        
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        in_features = 1280
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.backbone(x)


# -------------------------- Simple Dataset -------------------------- #

class SimpleFrameDataset(Dataset):
    """Simple dataset for evaluation."""
    
    def __init__(self, image_label_pairs: List[Tuple[str, int]], classes: List[str], img_size: int = 224):
        self.image_paths = [p for p, _ in image_label_pairs]
        self.labels = [l for _, l in image_label_pairs]
        self.classes = classes
        self.img_size = img_size
        
        self.transform = T.Compose([
            T.Resize((img_size, img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
    
    def __len__(self) -> int:
        return len(self.image_paths)
    
    def __getitem__(self, idx: int) -> Tuple[torch.Tensor, int, str]:
        image_path = self.image_paths[idx]
        label = self.labels[idx]
        
        try:
            img = Image.open(image_path).convert('RGB')
        except Exception as e:
            print(f"Warning: Failed to load {image_path}: {e}")
            img = Image.new('RGB', (self.img_size, self.img_size), color=(0, 0, 0))
        
        tensor = self.transform(img)
        return tensor, label, image_path


# -------------------------- Evaluation Functions -------------------------- #

def discover_frames(root_dir: str, split: str = 'test') -> Tuple[List[Tuple[str, int]], List[str]]:
    """Discover frames from dataset directory."""
    split_dir = os.path.join(root_dir, split)
    
    if not os.path.isdir(split_dir):
        raise ValueError(f"Split directory not found: {split_dir}")
    
    classes = sorted([d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))])
    
    if not classes:
        raise ValueError(f"No class directories found in {split_dir}")
    
    class_to_idx = {cls: idx for idx, cls in enumerate(classes)}
    
    image_label_pairs = []
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for class_name in classes:
        class_dir = os.path.join(split_dir, class_name)
        label = class_to_idx[class_name]
        
        for filename in os.listdir(class_dir):
            ext = os.path.splitext(filename)[1].lower()
            if ext in valid_extensions:
                image_path = os.path.join(class_dir, filename)
                image_label_pairs.append((image_path, label))
    
    return image_label_pairs, classes


def evaluate_model(
    model: nn.Module,
    loader: DataLoader,
    classes: List[str],
    device: torch.device
) -> Dict:
    """Comprehensive model evaluation."""
    model.eval()
    
    all_labels = []
    all_predictions = []
    all_probabilities = []
    all_paths = []
    
    print("Running evaluation...")
    with torch.no_grad():
        for inputs, labels, paths in tqdm(loader, desc="Evaluating"):
            inputs = inputs.to(device)
            labels = labels.to(device)
            
            outputs = model(inputs)
            probs = torch.softmax(outputs, dim=1)
            _, predicted = outputs.max(1)
            
            all_labels.extend(labels.cpu().tolist())
            all_predictions.extend(predicted.cpu().tolist())
            all_probabilities.extend(probs.cpu().tolist())
            all_paths.extend(paths)
    
    # Convert to numpy
    all_labels = np.array(all_labels)
    all_predictions = np.array(all_predictions)
    all_probabilities = np.array(all_probabilities)
    
    # Calculate metrics
    accuracy = 100 * (all_predictions == all_labels).sum() / len(all_labels)
    
    # Per-class accuracy
    class_accuracies = {}
    for i, cls in enumerate(classes):
        mask = all_labels == i
        if mask.sum() > 0:
            class_acc = 100 * (all_predictions[mask] == all_labels[mask]).sum() / mask.sum()
            class_accuracies[cls] = class_acc
    
    # Confusion matrix
    cm = confusion_matrix(all_labels, all_predictions)
    
    # Classification report
    report = classification_report(all_labels, all_predictions, target_names=classes, output_dict=True)
    
    # Confidence statistics
    confidence_stats = {}
    for i, cls in enumerate(classes):
        mask = all_labels == i
        if mask.sum() > 0:
            correct_mask = (all_predictions[mask] == all_labels[mask])
            incorrect_mask = ~correct_mask
            
            correct_confidences = all_probabilities[np.where(mask)[0][correct_mask], i]
            incorrect_confidences = all_probabilities[np.where(mask)[0][incorrect_mask], all_predictions[mask][incorrect_mask]]
            
            confidence_stats[cls] = {
                'correct_mean': float(np.mean(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'correct_std': float(np.std(correct_confidences)) if len(correct_confidences) > 0 else 0.0,
                'incorrect_mean': float(np.mean(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
                'incorrect_std': float(np.std(incorrect_confidences)) if len(incorrect_confidences) > 0 else 0.0,
            }
    
    # Find misclassified examples
    misclassified = []
    for i, (pred, true, path, probs) in enumerate(zip(all_predictions, all_labels, all_paths, all_probabilities)):
        if pred != true:
            misclassified.append({
                'path': path,
                'true_class': classes[true],
                'predicted_class': classes[pred],
                'confidence': float(probs[pred]),
                'true_class_prob': float(probs[true])
            })
    
    return {
        'overall_accuracy': accuracy,
        'class_accuracies': class_accuracies,
        'confusion_matrix': cm,
        'classification_report': report,
        'confidence_stats': confidence_stats,
        'misclassified': misclassified,
        'num_samples': len(all_labels)
    }


def plot_confusion_matrix(cm: np.ndarray, classes: List[str], save_path: str):
    """Plot confusion matrix."""
    plt.figure(figsize=(10, 8))
    
    # Normalize confusion matrix
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    
    sns.heatmap(
        cm_normalized,
        annot=True,
        fmt='.2%',
        cmap='Blues',
        xticklabels=classes,
        yticklabels=classes,
        cbar_kws={'label': 'Percentage'}
    )
    
    plt.title('Confusion Matrix (Normalized)', fontsize=14, fontweight='bold')
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"Confusion matrix saved to: {save_path}")
    plt.close()


def print_evaluation_report(results: Dict, classes: List[str]):
    """Print detailed evaluation report."""
    print("\n" + "=" * 70)
    print("EVALUATION REPORT")
    print("=" * 70)
    
    print(f"\nOverall Accuracy: {results['overall_accuracy']:.2f}%")
    print(f"Total Samples: {results['num_samples']}")
    
    print("\n" + "-" * 70)
    print("Per-Class Accuracy:")
    print("-" * 70)
    for cls, acc in results['class_accuracies'].items():
        print(f"  {cls:20s}: {acc:6.2f}%")
    
    print("\n" + "-" * 70)
    print("Confidence Statistics (for correctly/incorrectly classified samples):")
    print("-" * 70)
    for cls, stats in results['confidence_stats'].items():
        print(f"\n  {cls}:")
        print(f"    Correct   - Mean: {stats['correct_mean']:.3f}, Std: {stats['correct_std']:.3f}")
        print(f"    Incorrect - Mean: {stats['incorrect_mean']:.3f}, Std: {stats['incorrect_std']:.3f}")
    
    print("\n" + "-" * 70)
    print("Classification Report:")
    print("-" * 70)
    report = results['classification_report']
    print(f"{'Class':<20s} {'Precision':>10s} {'Recall':>10s} {'F1-Score':>10s} {'Support':>10s}")
    print("-" * 70)
    for cls in classes:
        if cls in report:
            r = report[cls]
            print(f"{cls:<20s} {r['precision']:>10.3f} {r['recall']:>10.3f} {r['f1-score']:>10.3f} {int(r['support']):>10d}")
    
    # Print most confident misclassifications
    print("\n" + "-" * 70)
    print("Top 10 Most Confident Misclassifications:")
    print("-" * 70)
    misclassified = sorted(results['misclassified'], key=lambda x: x['confidence'], reverse=True)[:10]
    for i, item in enumerate(misclassified, 1):
        print(f"\n{i}. {os.path.basename(item['path'])}")
        print(f"   True: {item['true_class']} | Predicted: {item['predicted_class']}")
        print(f"   Confidence: {item['confidence']:.3f} | True class prob: {item['true_class_prob']:.3f}")
    
    print("\n" + "=" * 70)


def main():
    """Main evaluation function."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Evaluate frame classifier')
    parser.add_argument('--model_dir', type=str, default='models/frame_classifier_mobilenet',
                        help='Model directory')
    parser.add_argument('--data_dir', type=str, default='datasets/behavior_frames',
                        help='Dataset directory')
    parser.add_argument('--split', type=str, default='test', choices=['train', 'valid', 'test'],
                        help='Dataset split to evaluate')
    parser.add_argument('--batch_size', type=int, default=32,
                        help='Batch size for evaluation')
    parser.add_argument('--save_dir', type=str, default='outputs/evaluation',
                        help='Directory to save evaluation results')
    
    args = parser.parse_args()
    
    # Create save directory
    os.makedirs(args.save_dir, exist_ok=True)
    
    # Load config
    config_path = os.path.join(args.model_dir, 'config.json')
    with open(config_path, 'r', encoding='utf-8') as f:
        config = json.load(f)
    
    # Load classes
    classes_path = os.path.join(args.model_dir, 'classes.json')
    with open(classes_path, 'r', encoding='utf-8') as f:
        classes = json.load(f)
    
    print(f"Classes: {classes}")
    
    # Set device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"Device: {device}")
    
    # Load model
    model = MobileNetV2Classifier(
        num_classes=len(classes),
        pretrained=False,
        dropout=config.get('dropout', 0.3)
    ).to(device)
    
    model_path = os.path.join(args.model_dir, 'best_model.pth')
    model.load_state_dict(torch.load(model_path, map_location=device, weights_only=True))
    model.eval()
    
    print(f"Model loaded from: {model_path}")
    
    # Discover frames
    image_label_pairs, discovered_classes = discover_frames(args.data_dir, args.split)
    print(f"\nFound {len(image_label_pairs)} images in {args.split} split")
    
    # Create dataset and loader
    dataset = SimpleFrameDataset(
        image_label_pairs,
        classes,
        img_size=config.get('img_size', 224)
    )
    
    loader = DataLoader(
        dataset,
        batch_size=args.batch_size,
        shuffle=False,
        num_workers=4,
        pin_memory=True
    )
    
    # Evaluate
    results = evaluate_model(model, loader, classes, device)
    
    # Print report
    print_evaluation_report(results, classes)
    
    # Plot confusion matrix
    cm_path = os.path.join(args.save_dir, f'confusion_matrix_{args.split}.png')
    plot_confusion_matrix(results['confusion_matrix'], classes, cm_path)
    
    # Save detailed results
    results_path = os.path.join(args.save_dir, f'evaluation_results_{args.split}.json')
    
    # Convert numpy arrays to lists for JSON serialization
    json_results = {
        'overall_accuracy': results['overall_accuracy'],
        'class_accuracies': results['class_accuracies'],
        'confusion_matrix': results['confusion_matrix'].tolist(),
        'classification_report': results['classification_report'],
        'confidence_stats': results['confidence_stats'],
        'misclassified': results['misclassified'][:50],  # Save top 50
        'num_samples': results['num_samples']
    }
    
    with open(results_path, 'w', encoding='utf-8') as f:
        json.dump(json_results, f, indent=2)
    
    print(f"\nDetailed results saved to: {results_path}")
    
    # Print data distribution issues
    print("\n" + "=" * 70)
    print("POTENTIAL ISSUES:")
    print("=" * 70)
    
    label_counts = Counter([label for _, label in image_label_pairs])
    print("\nClass distribution in dataset:")
    for cls_idx, cls_name in enumerate(classes):
        count = label_counts.get(cls_idx, 0)
        print(f"  {cls_name}: {count} samples")
    
    # Check for severe imbalance
    counts = list(label_counts.values())
    if counts:
        max_count = max(counts)
        min_count = min(counts)
        imbalance_ratio = max_count / max(min_count, 1)
        
        if imbalance_ratio > 5:
            print(f"\n⚠ WARNING: Severe class imbalance detected!")
            print(f"  Ratio: {imbalance_ratio:.1f}:1 (max:min)")
            print(f"  This can cause the model to be biased toward the majority class.")
    
    # Check for classes with low accuracy
    for cls, acc in results['class_accuracies'].items():
        if acc < 50:
            print(f"\n⚠ WARNING: Low accuracy for class '{cls}': {acc:.2f}%")
            print(f"  Consider collecting more diverse samples for this class.")


if __name__ == '__main__':
    main()
