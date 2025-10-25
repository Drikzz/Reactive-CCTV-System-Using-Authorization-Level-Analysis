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
            in_features = model.classifier[3].in_features
            
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
        
        # Save confusion matrix plot
        cm_path = save_dir / f"{self.model_name}_{self.mode}_confusion_matrix.png"
        self.plot_confusion_matrix(results['confusion_matrix'], cm_path)
        
        # Save metrics to JSON
        metrics_path = save_dir / f"{self.model_name}_{self.mode}_metrics.json"
        
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
        report_path = save_dir / f"{self.model_name}_{self.mode}_report.txt"
        with open(report_path, 'w') as f:
            # Redirect print to file
            import io
            from contextlib import redirect_stdout
            
            str_io = io.StringIO()
            with redirect_stdout(str_io):
                self.print_report(results)
            
            f.write(str_io.getvalue())
        
        print(f"✓ Text report saved to: {report_path}")


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
