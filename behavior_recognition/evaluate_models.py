"""
Model Evaluation Script - Confusion Matrix and Performance Metrics

This script evaluates trained models (both transfer learning and feature extraction)
on the test dataset and generates:
1. Confusion matrix visualization
2. Classification metrics: Precision, Recall, F1-Score, Accuracy
3. Per-class performance breakdown
4. Detailed evaluation report

Supports:
- MobileNetV2
- MobileNetV3-Small
- ShuffleNetV2

Both training modes:
- Transfer Learning (fine-tuned backbone)
- Feature Extraction (frozen backbone)

Usage:
    # Evaluate a single model
    python evaluate_models.py --model mobilenetv2 --mode feature_extraction
    
    # Evaluate all models and modes
    python evaluate_models.py --model all --mode both
    
    # Save detailed results
    python evaluate_models.py --model mobilenetv2 --mode both --save-dir evaluation_results
"""

import os
import sys
from pathlib import Path
import argparse
import json
from collections import defaultdict

import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader
from torchvision import datasets, transforms, models
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    confusion_matrix, 
    precision_score, 
    recall_score, 
    f1_score, 
    accuracy_score,
    classification_report
)
from tqdm import tqdm

# Add project root to path
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


class ModelEvaluator:
    """Evaluate trained models on test dataset with confusion matrix and metrics."""
    
    def __init__(self, model_name: str, mode: str, device: str = None):
        """
        Initialize evaluator.
        
        Args:
            model_name: Model architecture (mobilenetv2, mobilenetv3, shufflenetv2)
            mode: Training mode (transfer or feature_extraction)
            device: Device to use (cuda or cpu)
        """
        self.model_name = model_name.lower()
        self.mode = mode
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Define model configurations
        self.model_configs = {
            'mobilenetv2': {
                'transfer': 'models/mobilenetv2/mobilenet_transfer.pth',
                'feature_extraction': 'models/mobilenetv2/mobilenet_feature_extraction.pth'
            },
            'mobilenetv3': {
                'transfer': 'models/mobilenetv3-small/mobilenet_v3_small_transfer.pth',
                'feature_extraction': 'models/mobilenetv3-small/mobilenetv3_small_feature_extraction.pth'
            },
            'shufflenetv2': {
                'transfer': 'models/shufflenetv2/shufflenet_v2_transfer.pth',
                'feature_extraction': 'models/shufflenetv2/shufflenetv2_feature_extraction.pth'
            }
        }
        
        # Test dataset path
        self.test_dir = BASE_DIR / 'datasets' / 'test'
        
        # Load model
        self.model, self.class_names, self.num_classes = self._load_model()
        
        # Prepare test data
        self.test_loader = self._prepare_test_data()
        
        print(f"\n{'='*70}")
        print(f"Model Evaluation: {model_name.upper()} ({mode})")
        print(f"{'='*70}")
        print(f"Device: {self.device}")
        print(f"Classes: {self.class_names}")
        print(f"Test Samples: {len(self.test_loader.dataset)}")
        print(f"{'='*70}\n")
    
    def _load_model(self):
        """Load trained model from checkpoint."""
        model_path = BASE_DIR / self.model_configs[self.model_name][self.mode]
        
        if not model_path.exists():
            raise FileNotFoundError(f"Model not found: {model_path}")
        
        print(f"Loading model from: {model_path}")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        # Get model info
        num_classes = checkpoint['num_classes']
        class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(num_classes)])
        
        # Build model architecture
        if self.model_name == 'mobilenetv2':
            model = models.mobilenet_v2(weights=None)
            in_features = model.classifier[1].in_features
            
            # Check if feature extraction model (has Sequential classifier)
            if 'classifier.1.weight' in checkpoint['model_state_dict']:
                # Feature extraction model
                model.classifier = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features, num_classes)
                )
            else:
                # Transfer learning model
                model.classifier[1] = nn.Linear(in_features, num_classes)
                
        elif self.model_name == 'mobilenetv3':
            model = models.mobilenet_v3_small(weights=None)
            # MobileNetV3-Small's feature extractor outputs 576 features
            in_features = 576
            
            # Check if feature extraction model
            if 'classifier.3.weight' in checkpoint['model_state_dict']:
                # Feature extraction model
                model.classifier = nn.Sequential(
                    nn.Linear(in_features, 1024),
                    nn.Hardswish(),
                    nn.Dropout(p=0.2),
                    nn.Linear(1024, num_classes)
                )
            else:
                # Transfer learning model
                model.classifier[3] = nn.Linear(in_features, num_classes)
                
        elif self.model_name == 'shufflenetv2':
            model = models.shufflenet_v2_x1_0(weights=None)
            in_features = model.fc.in_features
            
            # Check if feature extraction model
            if 'fc.1.weight' in checkpoint['model_state_dict']:
                # Feature extraction model
                model.fc = nn.Sequential(
                    nn.Dropout(p=0.2),
                    nn.Linear(in_features, num_classes)
                )
            else:
                # Transfer learning model
                model.fc = nn.Linear(in_features, num_classes)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        print(f"✓ Loaded {self.model_name} ({self.mode}) with {num_classes} classes")
        
        return model, class_names, num_classes
    
    def _prepare_test_data(self):
        """Prepare test dataset loader."""
        transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        test_dataset = datasets.ImageFolder(self.test_dir, transform=transform)
        test_loader = DataLoader(test_dataset, batch_size=32, shuffle=False, num_workers=2)
        
        return test_loader
    
    def evaluate(self):
        """
        Run evaluation on test dataset.
        
        Returns:
            dict: Dictionary containing predictions, true labels, and metrics
        """
        print("Running evaluation on test dataset...")
        
        all_preds = []
        all_labels = []
        all_probs = []
        
        with torch.no_grad():
            for images, labels in tqdm(self.test_loader, desc="Evaluating"):
                images = images.to(self.device)
                
                # Forward pass
                outputs = self.model(images)
                probabilities = torch.nn.functional.softmax(outputs, dim=1)
                _, predicted = torch.max(outputs, 1)
                
                # Store results
                all_preds.extend(predicted.cpu().numpy())
                all_labels.extend(labels.numpy())
                all_probs.extend(probabilities.cpu().numpy())
        
        # Convert to numpy arrays
        all_preds = np.array(all_preds)
        all_labels = np.array(all_labels)
        all_probs = np.array(all_probs)
        
        # Calculate metrics
        metrics = self._calculate_metrics(all_labels, all_preds)
        
        # Generate confusion matrix
        cm = confusion_matrix(all_labels, all_preds)
        
        results = {
            'predictions': all_preds,
            'true_labels': all_labels,
            'probabilities': all_probs,
            'confusion_matrix': cm,
            'metrics': metrics,
            'class_names': self.class_names
        }
        
        return results
    
    def _calculate_metrics(self, y_true, y_pred):
        """
        Calculate all evaluation metrics.
        
        Args:
            y_true: True labels
            y_pred: Predicted labels
            
        Returns:
            dict: Dictionary of metrics
        """
        # Overall metrics
        accuracy = accuracy_score(y_true, y_pred)
        
        # Per-class metrics (macro average)
        precision_macro = precision_score(y_true, y_pred, average='macro', zero_division=0)
        recall_macro = recall_score(y_true, y_pred, average='macro', zero_division=0)
        f1_macro = f1_score(y_true, y_pred, average='macro', zero_division=0)
        
        # Per-class metrics (weighted average)
        precision_weighted = precision_score(y_true, y_pred, average='weighted', zero_division=0)
        recall_weighted = recall_score(y_true, y_pred, average='weighted', zero_division=0)
        f1_weighted = f1_score(y_true, y_pred, average='weighted', zero_division=0)
        
        # Per-class metrics (individual)
        precision_per_class = precision_score(y_true, y_pred, average=None, zero_division=0)
        recall_per_class = recall_score(y_true, y_pred, average=None, zero_division=0)
        f1_per_class = f1_score(y_true, y_pred, average=None, zero_division=0)
        
        metrics = {
            'accuracy': accuracy,
            'precision_macro': precision_macro,
            'recall_macro': recall_macro,
            'f1_macro': f1_macro,
            'precision_weighted': precision_weighted,
            'recall_weighted': recall_weighted,
            'f1_weighted': f1_weighted,
            'per_class': {
                'precision': precision_per_class.tolist(),
                'recall': recall_per_class.tolist(),
                'f1': f1_per_class.tolist()
            }
        }
        
        return metrics
    
    def plot_confusion_matrix(self, cm, save_path=None, title=None):
        """
        Plot confusion matrix as a heatmap.
        
        Args:
            cm: Confusion matrix
            save_path: Path to save the plot
            title: Plot title
        """
        plt.figure(figsize=(10, 8))
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', 
                   xticklabels=self.class_names,
                   yticklabels=self.class_names,
                   cbar_kws={'label': 'Count'})
        
        plt.ylabel('True Label')
        plt.xlabel('Predicted Label')
        
        if title:
            plt.title(title)
        else:
            plt.title(f'Confusion Matrix - {self.model_name.upper()} ({self.mode})')
        
        plt.tight_layout()
        
        if save_path:
            plt.savefig(save_path, dpi=300, bbox_inches='tight')
            print(f"✓ Confusion matrix saved to: {save_path}")
        
        plt.close()
    
    def print_report(self, results):
        """Print detailed evaluation report."""
        metrics = results['metrics']
        cm = results['confusion_matrix']
        
        print("\n" + "="*70)
        print(f"EVALUATION REPORT: {self.model_name.upper()} ({self.mode})")
        print("="*70)
        
        # Overall metrics
        print("\n📊 OVERALL METRICS:")
        print("-"*70)
        print(f"Accuracy:           {metrics['accuracy']*100:>6.2f}%")
        print(f"Precision (Macro):  {metrics['precision_macro']*100:>6.2f}%")
        print(f"Recall (Macro):     {metrics['recall_macro']*100:>6.2f}%")
        print(f"F1-Score (Macro):   {metrics['f1_macro']*100:>6.2f}%")
        print()
        print(f"Precision (Weighted): {metrics['precision_weighted']*100:>6.2f}%")
        print(f"Recall (Weighted):    {metrics['recall_weighted']*100:>6.2f}%")
        print(f"F1-Score (Weighted):  {metrics['f1_weighted']*100:>6.2f}%")
        
        # Per-class metrics
        print("\n📋 PER-CLASS METRICS:")
        print("-"*70)
        print(f"{'Class':<25} | {'Precision':<10} | {'Recall':<10} | {'F1-Score':<10}")
        print("-"*70)
        
        for i, class_name in enumerate(self.class_names):
            precision = metrics['per_class']['precision'][i] * 100
            recall = metrics['per_class']['recall'][i] * 100
            f1 = metrics['per_class']['f1'][i] * 100
            print(f"{class_name:<25} | {precision:>9.2f}% | {recall:>9.2f}% | {f1:>9.2f}%")
        
        # Confusion matrix
        print("\n🔢 CONFUSION MATRIX:")
        print("-"*70)
        
        # Header
        header = f"{'True \\ Pred':<20}"
        for class_name in self.class_names:
            header += f"{class_name[:15]:>16}"
        print(header)
        print("-"*70)
        
        # Rows
        for i, true_class in enumerate(self.class_names):
            row = f"{true_class:<20}"
            for j in range(len(self.class_names)):
                row += f"{cm[i][j]:>16}"
            print(row)
        
        print("\n" + "="*70 + "\n")
    
    def save_results(self, results, save_dir):
        """
        Save evaluation results to files.
        
        Args:
            results: Evaluation results dictionary
            save_dir: Directory to save results
        """
        save_dir = Path(save_dir)
        save_dir.mkdir(parents=True, exist_ok=True)
        
        # Create organized folder structure: save_dir/model_name/mode/
        model_folder = save_dir / self.model_name / self.mode
        model_folder.mkdir(parents=True, exist_ok=True)
        
        # Save confusion matrix plot
        cm_path = model_folder / "confusion_matrix.png"
        self.plot_confusion_matrix(results['confusion_matrix'], cm_path)
        
        # Save metrics to JSON
        metrics_path = model_folder / "metrics.json"
        
        # Convert numpy arrays to lists for JSON serialization
        serializable_metrics = {
            'model': self.model_name,
            'mode': self.mode,
            'accuracy': float(results['metrics']['accuracy']),
            'precision_macro': float(results['metrics']['precision_macro']),
            'recall_macro': float(results['metrics']['recall_macro']),
            'f1_macro': float(results['metrics']['f1_macro']),
            'precision_weighted': float(results['metrics']['precision_weighted']),
            'recall_weighted': float(results['metrics']['recall_weighted']),
            'f1_weighted': float(results['metrics']['f1_weighted']),
            'per_class_metrics': {
                class_name: {
                    'precision': float(results['metrics']['per_class']['precision'][i]),
                    'recall': float(results['metrics']['per_class']['recall'][i]),
                    'f1': float(results['metrics']['per_class']['f1'][i])
                }
                for i, class_name in enumerate(self.class_names)
            },
            'confusion_matrix': results['confusion_matrix'].tolist(),
            'class_names': self.class_names
        }
        
        with open(metrics_path, 'w') as f:
            json.dump(serializable_metrics, f, indent=2)
        
        print(f"✓ Metrics saved to: {metrics_path}")
        
        # Save text report
        report_path = model_folder / "report.txt"
        with open(report_path, 'w') as f:
            # Redirect print to file
            import io
            from contextlib import redirect_stdout
            
            str_io = io.StringIO()
            with redirect_stdout(str_io):
                self.print_report(results)
            
            f.write(str_io.getvalue())
        
        print(f"✓ Text report saved to: {report_path}")


def plot_combined_confusion_matrices(results_dict, class_names, save_path, title="Combined Confusion Matrices"):
    """
    Plot multiple confusion matrices in a single figure.
    
    Args:
        results_dict: Dictionary of {model_name: results}
        class_names: List of class names
        save_path: Path to save the combined plot
        title: Overall title for the figure
    """
    n_models = len(results_dict)
    
    # Determine grid layout
    if n_models == 1:
        rows, cols = 1, 1
        figsize = (8, 6)
    elif n_models == 2:
        rows, cols = 1, 2
        figsize = (16, 6)
    elif n_models == 3:
        rows, cols = 1, 3
        figsize = (18, 5)
    elif n_models == 4:
        rows, cols = 2, 2
        figsize = (14, 12)
    elif n_models <= 6:
        rows, cols = 2, 3
        figsize = (18, 10)
    else:
        rows, cols = 3, 3
        figsize = (18, 15)
    
    fig, axes = plt.subplots(rows, cols, figsize=figsize)
    
    # Flatten axes for easier iteration
    if n_models == 1:
        axes = [axes]
    else:
        axes = axes.flatten()
    
    # Plot each confusion matrix
    for idx, (model_key, results) in enumerate(results_dict.items()):
        ax = axes[idx]
        cm = results['confusion_matrix']
        
        # Create heatmap
        sns.heatmap(cm, annot=True, fmt='d', cmap='Blues',
                   xticklabels=class_names,
                   yticklabels=class_names,
                   ax=ax,
                   cbar=True,
                   square=True)
        
        # Format model name for display
        model_name, mode = model_key.rsplit('_', 1)
        display_name = f"{model_name.upper()}\n({mode.replace('_', ' ').title()})"
        
        ax.set_title(display_name, fontsize=10, fontweight='bold', pad=10)  # Added padding
        ax.set_ylabel('True Label', fontsize=9)
        ax.set_xlabel('Predicted Label', fontsize=9, labelpad=8)  # Added padding
        
        # Rotate labels for readability
        ax.set_xticklabels(ax.get_xticklabels(), rotation=45, ha='right', fontsize=8)
        ax.set_yticklabels(ax.get_yticklabels(), rotation=0, fontsize=8)
    
    # Hide unused subplots
    for idx in range(n_models, len(axes)):
        axes[idx].axis('off')
    
    # Overall title - positioned higher to avoid overlap
    fig.suptitle(title, fontsize=14, fontweight='bold', y=0.995)
    
    # Add padding between subplots and leave space for suptitle
    plt.tight_layout(h_pad=3.0, w_pad=2.0, rect=[0, 0, 1, 0.97])
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Combined confusion matrices saved to: {save_path}")
    plt.close()


def plot_metrics_comparison(results_dict, save_path, title="Model Performance Comparison"):
    """
    Plot bar chart comparing metrics across models.
    
    Args:
        results_dict: Dictionary of {model_name: results}
        save_path: Path to save the comparison plot
        title: Overall title for the figure
    """
    # Extract metrics
    model_names = []
    accuracies = []
    precisions = []
    recalls = []
    f1_scores = []
    
    for model_key, results in results_dict.items():
        metrics = results['metrics']
        
        # Format model name
        model_name, mode = model_key.rsplit('_', 1)
        display_name = f"{model_name}\n({mode[:4]})"  # Abbreviated mode
        
        model_names.append(display_name)
        accuracies.append(metrics['accuracy'] * 100)
        precisions.append(metrics['precision_macro'] * 100)
        recalls.append(metrics['recall_macro'] * 100)
        f1_scores.append(metrics['f1_macro'] * 100)
    
    # Create bar chart
    x = np.arange(len(model_names))
    width = 0.2
    
    fig, ax = plt.subplots(figsize=(max(12, len(model_names) * 1.5), 6))
    
    bars1 = ax.bar(x - 1.5*width, accuracies, width, label='Accuracy', color='#2E86AB')
    bars2 = ax.bar(x - 0.5*width, precisions, width, label='Precision', color='#A23B72')
    bars3 = ax.bar(x + 0.5*width, recalls, width, label='Recall', color='#F18F01')
    bars4 = ax.bar(x + 1.5*width, f1_scores, width, label='F1-Score', color='#C73E1D')
    
    # Add value labels on bars
    def add_labels(bars):
        for bar in bars:
            height = bar.get_height()
            ax.text(bar.get_x() + bar.get_width()/2., height,
                   f'{height:.1f}%',
                   ha='center', va='bottom', fontsize=8)
    
    add_labels(bars1)
    add_labels(bars2)
    add_labels(bars3)
    add_labels(bars4)
    
    ax.set_xlabel('Model', fontsize=12, fontweight='bold')
    ax.set_ylabel('Score (%)', fontsize=12, fontweight='bold')
    ax.set_title(title, fontsize=14, fontweight='bold')
    ax.set_xticks(x)
    ax.set_xticklabels(model_names, fontsize=9)
    ax.legend(loc='lower right', fontsize=10)
    ax.set_ylim([0, 105])
    ax.grid(axis='y', alpha=0.3, linestyle='--')
    
    plt.tight_layout()
    plt.savefig(save_path, dpi=300, bbox_inches='tight')
    print(f"✓ Metrics comparison saved to: {save_path}")
    plt.close()


def main():
    parser = argparse.ArgumentParser(
        description='Evaluate trained models on test dataset with confusion matrix and metrics'
    )
    
    parser.add_argument('--model', type=str, 
                       choices=['mobilenetv2', 'mobilenetv3', 'shufflenetv2', 'all'],
                       default='mobilenetv2',
                       help='Model to evaluate (default: mobilenetv2)')
    parser.add_argument('--mode', type=str,
                       choices=['transfer', 'feature_extraction', 'both'],
                       default='feature_extraction',
                       help='Training mode to evaluate (default: feature_extraction)')
    parser.add_argument('--save-dir', type=str, default='evaluation_results',
                       help='Directory to save results (default: evaluation_results)')
    parser.add_argument('--device', type=str, default=None,
                       help='Device to use (cuda or cpu, default: auto-detect)')
    
    args = parser.parse_args()
    
    # Determine which models to evaluate
    if args.model == 'all':
        models_to_eval = ['mobilenetv2', 'mobilenetv3', 'shufflenetv2']
    else:
        models_to_eval = [args.model]
    
    # Determine which modes to evaluate
    if args.mode == 'both':
        modes_to_eval = ['transfer', 'feature_extraction']
    else:
        modes_to_eval = [args.mode]
    
    # Run evaluations
    all_results = {}
    
    for model_name in models_to_eval:
        for mode in modes_to_eval:
            try:
                print(f"\n{'#'*70}")
                print(f"# Evaluating: {model_name.upper()} ({mode})")
                print(f"{'#'*70}\n")
                
                # Create evaluator
                evaluator = ModelEvaluator(
                    model_name=model_name,
                    mode=mode,
                    device=args.device
                )
                
                # Run evaluation
                results = evaluator.evaluate()
                
                # Print report
                evaluator.print_report(results)
                
                # Save results
                evaluator.save_results(results, args.save_dir)
                
                # Store results
                key = f"{model_name}_{mode}"
                all_results[key] = results
                
            except Exception as e:
                print(f"❌ Error evaluating {model_name} ({mode}): {e}")
                import traceback
                traceback.print_exc()
    
    # Print summary
    print("\n" + "="*70)
    print("EVALUATION COMPLETE")
    print("="*70)
    print(f"\nResults saved to: {args.save_dir}/")
    print(f"Total evaluations: {len(all_results)}")
    
    # Generate combined visualizations if multiple models evaluated
    if len(all_results) > 0:
        save_dir = Path(args.save_dir)
        
        # Get class names (same for all models)
        first_result = next(iter(all_results.values()))
        class_names = first_result['class_names']
        
        # 1. Combined confusion matrix for ALL models (if evaluating multiple)
        if len(all_results) >= 2:
            print(f"\n📊 Generating combined visualizations...")
            
            combined_cm_path = save_dir / "all_models_confusion_matrices.png"
            plot_combined_confusion_matrices(
                all_results, 
                class_names,
                combined_cm_path,
                title="Confusion Matrices - All Models"
            )
            
            # Metrics comparison chart
            metrics_comparison_path = save_dir / "all_models_metrics_comparison.png"
            plot_metrics_comparison(
                all_results,
                metrics_comparison_path,
                title="Performance Comparison - All Models"
            )
        
        # 2. Feature Extraction models only (if 3 or more FE models)
        fe_results = {k: v for k, v in all_results.items() if 'feature_extraction' in k}
        if len(fe_results) >= 2:
            fe_cm_path = save_dir / "feature_extraction_confusion_matrices.png"
            plot_combined_confusion_matrices(
                fe_results,
                class_names,
                fe_cm_path,
                title="Confusion Matrices - Feature Extraction Models"
            )
            
            fe_metrics_path = save_dir / "feature_extraction_metrics_comparison.png"
            plot_metrics_comparison(
                fe_results,
                fe_metrics_path,
                title="Performance Comparison - Feature Extraction Models"
            )
        
        # 3. Transfer Learning models only (if 3 or more TL models)
        tl_results = {k: v for k, v in all_results.items() if 'transfer' in k and 'feature_extraction' not in k}
        if len(tl_results) >= 2:
            tl_cm_path = save_dir / "transfer_learning_confusion_matrices.png"
            plot_combined_confusion_matrices(
                tl_results,
                class_names,
                tl_cm_path,
                title="Confusion Matrices - Transfer Learning Models"
            )
            
            tl_metrics_path = save_dir / "transfer_learning_metrics_comparison.png"
            plot_metrics_comparison(
                tl_results,
                tl_metrics_path,
                title="Performance Comparison - Transfer Learning Models"
            )
    
    # Print comparison table if multiple models evaluated
    if len(all_results) > 1:
        print("\n📊 COMPARISON SUMMARY:")
        print("-"*70)
        print(f"{'Model':<30} | {'Accuracy':<10} | {'F1-Macro':<10}")
        print("-"*70)
        
        for key, results in all_results.items():
            accuracy = results['metrics']['accuracy'] * 100
            f1 = results['metrics']['f1_macro'] * 100
            print(f"{key:<30} | {accuracy:>9.2f}% | {f1:>9.2f}%")
        
        print("-"*70)
    
    print("\n✓ All evaluations completed successfully!\n")


if __name__ == '__main__':
    main()
