"""
Compare Training Metrics Across Models

This script reads and compares training summary files from MobileNetV2, MobileNetV3-Small, 
and ShuffleNetV2 models. It supports comparing both transfer learning (fine-tuning) and 
feature extraction (frozen backbone) training modes. Generates comprehensive comparison 
tables and saves them as both text and markdown files suitable for thesis inclusion.

Usage:
    # Compare transfer learning (default)
    python compare_training_metrics.py
    
    # Compare feature extraction
    python compare_training_metrics.py --mode feature_extraction
    
    # Compare both modes (6 models total)
    python compare_training_metrics.py --mode both
    
    # Custom output file
    python compare_training_metrics.py --save my_comparison.md
"""

import argparse
import re
from pathlib import Path
from typing import Dict, Optional
import json


def parse_training_summary(file_path: Path) -> Optional[Dict]:
    """
    Parse a training summary file and extract key metrics.
    
    Args:
        file_path: Path to the training summary text file
        
    Returns:
        Dictionary containing extracted metrics or None if file doesn't exist
    """
    if not file_path.exists():
        return None
    
    with open(file_path, 'r') as f:
        content = f.read()
    
    metrics = {}
    
    # Extract metrics using regex
    patterns = {
        'total_epochs': r'Total Epochs:\s*(\d+)',
        'best_val_acc': r'Best Val Acc:\s*([\d.]+)%\s*\(Epoch\s*(\d+)\)',
        'final_train_loss': r'Final Train Loss:\s*([\d.]+)',
        'final_val_loss': r'Final Val Loss:\s*([\d.]+)',
        'final_train_acc': r'Final Train Acc:\s*([\d.]+)%',
        'final_val_acc': r'Final Val Acc:\s*([\d.]+)%',
        'train_val_gap': r'Train-Val Gap:\s*([\d.]+)%',
        'status': r'Status:\s*(.+)',
        'overall_test_acc': r'Overall Test Accuracy:\s*([\d.]+)%',
        'training_time': r'Total training time:\s*([\d.]+)\s*minutes'
    }
    
    for key, pattern in patterns.items():
        match = re.search(pattern, content)
        if match:
            if key == 'best_val_acc':
                metrics['best_val_acc'] = float(match.group(1))
                metrics['best_epoch'] = int(match.group(2))
            elif key == 'status':
                metrics[key] = match.group(1).strip()
            else:
                try:
                    metrics[key] = float(match.group(1))
                except ValueError:
                    metrics[key] = match.group(1)
    
    # Extract per-class test accuracy
    per_class_pattern = r'Per-class test accuracy:\n((?:\s+.+:.+\n)+)'
    per_class_match = re.search(per_class_pattern, content)
    if per_class_match:
        per_class_str = per_class_match.group(1)
        per_class_lines = per_class_str.strip().split('\n')
        metrics['per_class_test'] = {}
        for line in per_class_lines:
            # Format: "  class_name: XX.X% (correct/total)"
            match = re.match(r'\s+(.+):\s*([\d.]+)%\s*\((\d+)/(\d+)\)', line)
            if match:
                class_name = match.group(1).strip()
                accuracy = float(match.group(2))
                correct = int(match.group(3))
                total = int(match.group(4))
                metrics['per_class_test'][class_name] = {
                    'accuracy': accuracy,
                    'correct': correct,
                    'total': total
                }
    
    return metrics


def format_comparison_table(metrics_dict: Dict[str, Dict]) -> str:
    """
    Format comparison metrics into a readable table.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        
    Returns:
        Formatted string table
    """
    output = []
    output.append("="*100)
    output.append("MODEL TRAINING METRICS COMPARISON")
    output.append("="*100)
    output.append("")
    
    # Model names
    models = list(metrics_dict.keys())
    
    # Overall Metrics Table
    output.append("OVERALL TRAINING METRICS")
    output.append("-"*100)
    
    # For 6 models, split into two tables (3 models each) for better readability
    if len(models) > 3:
        # First table - first 3 models
        output.append(f"{'Metric':<30} | {models[0]:<25} | {models[1]:<25} | {models[2]:<25}")
        output.append("-"*100)
    else:
        # Original format for 3 or fewer models
        if len(models) == 3:
            output.append(f"{'Metric':<30} | {models[0]:<20} | {models[1]:<20} | {models[2]:<20}")
        elif len(models) == 2:
            output.append(f"{'Metric':<30} | {models[0]:<20} | {models[1]:<20}")
        else:
            output.append(f"{'Metric':<30} | {models[0]:<20}")
        output.append("-"*100)
    
    # Metrics to compare
    metric_labels = [
        ('total_epochs', 'Total Epochs', ''),
        ('best_val_acc', 'Best Validation Acc', '%'),
        ('best_epoch', 'Best Epoch', ''),
        ('final_train_acc', 'Final Train Acc', '%'),
        ('final_val_acc', 'Final Val Acc', '%'),
        ('train_val_gap', 'Train-Val Gap', '%'),
        ('overall_test_acc', 'Overall Test Acc', '%'),
        ('final_train_loss', 'Final Train Loss', ''),
        ('final_val_loss', 'Final Val Loss', ''),
        ('training_time', 'Training Time (min)', ''),
    ]
    
    for key, label, unit in metric_labels:
        values = []
        for model in models:
            if key in metrics_dict[model]:
                val = metrics_dict[model][key]
                if isinstance(val, float):
                    if 'loss' in key.lower():
                        values.append(f"{val:.4f}{unit}")
                    elif 'time' in key.lower():
                        values.append(f"{val:.2f}{unit}")
                    else:
                        values.append(f"{val:.2f}{unit}")
                else:
                    values.append(f"{val}{unit}")
            else:
                values.append("N/A")
        
        # Build rows based on number of models
        if len(models) > 3:
            # Split into two rows for 6 models
            row1 = f"{label:<30} | {values[0]:<25} | {values[1]:<25} | {values[2]:<25}"
            output.append(row1)
        else:
            # Original single row for 3 or fewer models
            if len(models) == 3:
                output.append(f"{label:<30} | {values[0]:<20} | {values[1]:<20} | {values[2]:<20}")
            elif len(models) == 2:
                output.append(f"{label:<30} | {values[0]:<20} | {values[1]:<20}")
            else:
                output.append(f"{label:<30} | {values[0]:<20}")
    
    # If more than 3 models, add second table
    if len(models) > 3:
        output.append("")
        output.append("-"*100)
        output.append(f"{'Metric':<30} | {models[3]:<25} | {models[4]:<25} | {models[5]:<25}")
        output.append("-"*100)
        
        for key, label, unit in metric_labels:
            values = []
            for model in models[3:]:  # Get remaining models
                if key in metrics_dict[model]:
                    val = metrics_dict[model][key]
                    if isinstance(val, float):
                        if 'loss' in key.lower():
                            values.append(f"{val:.4f}{unit}")
                        elif 'time' in key.lower():
                            values.append(f"{val:.2f}{unit}")
                        else:
                            values.append(f"{val:.2f}{unit}")
                    else:
                        values.append(f"{val}{unit}")
                else:
                    values.append("N/A")
            
            row2 = f"{label:<30} | {values[0]:<25} | {values[1]:<25} | {values[2]:<25}"
            output.append(row2)
    
    # Status
    output.append("-"*100)
    
    # Split status for 6 models
    if len(models) > 3:
        status_line1 = f"{'Overfitting Status':<30} | "
        for i in range(3):
            status = metrics_dict[models[i]].get('status', 'N/A')
            status_line1 += f"{status:<25}"
            if i < 2:
                status_line1 += " | "
        output.append(status_line1)
        output.append("")
        output.append("-"*100)
        
        status_line2 = f"{'Overfitting Status':<30} | "
        for i in range(3, 6):
            status = metrics_dict[models[i]].get('status', 'N/A')
            status_line2 += f"{status:<25}"
            if i < 5:
                status_line2 += " | "
        output.append(status_line2)
    else:
        status_line = f"{'Overfitting Status':<30} | "
        for i, model in enumerate(models):
            status = metrics_dict[model].get('status', 'N/A')
            status_line += f"{status:<20}"
            if i < len(models) - 1:
                status_line += " | "
        output.append(status_line)
    
    output.append("")
    output.append("")
    
    # Per-Class Test Accuracy Comparison
    output.append("PER-CLASS TEST ACCURACY")
    output.append("-"*100)
    
    # Get all class names
    all_classes = set()
    for model in models:
        if 'per_class_test' in metrics_dict[model]:
            all_classes.update(metrics_dict[model]['per_class_test'].keys())
    
    if all_classes:
        # Split into two tables for 6 models
        if len(models) > 3:
            output.append(f"{'Class':<25} | {models[0]:<25} | {models[1]:<25} | {models[2]:<25}")
            output.append("-"*100)
            
            for class_name in sorted(all_classes):
                values = []
                for model in models[:3]:
                    if 'per_class_test' in metrics_dict[model] and class_name in metrics_dict[model]['per_class_test']:
                        acc = metrics_dict[model]['per_class_test'][class_name]['accuracy']
                        correct = metrics_dict[model]['per_class_test'][class_name]['correct']
                        total = metrics_dict[model]['per_class_test'][class_name]['total']
                        values.append(f"{acc:.1f}% ({correct}/{total})")
                    else:
                        values.append("N/A")
                
                output.append(f"{class_name:<25} | {values[0]:<25} | {values[1]:<25} | {values[2]:<25}")
            
            output.append("")
            output.append("-"*100)
            output.append(f"{'Class':<25} | {models[3]:<25} | {models[4]:<25} | {models[5]:<25}")
            output.append("-"*100)
            
            for class_name in sorted(all_classes):
                values = []
                for model in models[3:]:
                    if 'per_class_test' in metrics_dict[model] and class_name in metrics_dict[model]['per_class_test']:
                        acc = metrics_dict[model]['per_class_test'][class_name]['accuracy']
                        correct = metrics_dict[model]['per_class_test'][class_name]['correct']
                        total = metrics_dict[model]['per_class_test'][class_name]['total']
                        values.append(f"{acc:.1f}% ({correct}/{total})")
                    else:
                        values.append("N/A")
                
                output.append(f"{class_name:<25} | {values[0]:<25} | {values[1]:<25} | {values[2]:<25}")
        else:
            # Original format for 3 or fewer models
            output.append(f"{'Class':<25} | {models[0]:<20} | {models[1]:<20} | {models[2]:<20}")
            output.append("-"*100)
            
            for class_name in sorted(all_classes):
                values = []
                for model in models:
                    if 'per_class_test' in metrics_dict[model] and class_name in metrics_dict[model]['per_class_test']:
                        acc = metrics_dict[model]['per_class_test'][class_name]['accuracy']
                        correct = metrics_dict[model]['per_class_test'][class_name]['correct']
                        total = metrics_dict[model]['per_class_test'][class_name]['total']
                        values.append(f"{acc:.1f}% ({correct}/{total})")
                    else:
                        values.append("N/A")
                
                output.append(f"{class_name:<25} | {values[0]:<20} | {values[1]:<20} | {values[2]:<20}")
    
    output.append("-"*100)
    output.append("")
    
    # Summary Analysis
    output.append("SUMMARY ANALYSIS")
    output.append("-"*100)
    
    # Find best model for each metric
    test_accs = {model: metrics_dict[model].get('overall_test_acc', 0) for model in models}
    best_accuracy_model = max(test_accs, key=test_accs.get)
    
    train_times = {model: metrics_dict[model].get('training_time', float('inf')) for model in models}
    fastest_model = min(train_times, key=train_times.get)
    
    gaps = {model: metrics_dict[model].get('train_val_gap', float('inf')) for model in models}
    least_overfit_model = min(gaps, key=gaps.get)
    
    output.append(f"🏆 Best Test Accuracy: {best_accuracy_model} ({test_accs[best_accuracy_model]:.2f}%)")
    output.append(f"⚡ Fastest Training: {fastest_model} ({train_times[fastest_model]:.2f} min)")
    output.append(f"✅ Least Overfitting: {least_overfit_model} (gap: {gaps[least_overfit_model]:.2f}%)")
    output.append("")
    
    # Accuracy rankings
    sorted_by_acc = sorted(test_accs.items(), key=lambda x: x[1], reverse=True)
    output.append("Accuracy Ranking:")
    for rank, (model, acc) in enumerate(sorted_by_acc, 1):
        output.append(f"  {rank}. {model}: {acc:.2f}%")
    
    output.append("")
    
    # Speed rankings
    sorted_by_speed = sorted(train_times.items(), key=lambda x: x[1])
    output.append("Training Speed Ranking:")
    for rank, (model, time) in enumerate(sorted_by_speed, 1):
        if time != float('inf'):
            output.append(f"  {rank}. {model}: {time:.2f} min")
    
    output.append("="*100)
    
    return '\n'.join(output)


def format_markdown_table(metrics_dict: Dict[str, Dict]) -> str:
    """
    Format comparison metrics as a markdown table for thesis.
    
    Args:
        metrics_dict: Dictionary mapping model names to their metrics
        
    Returns:
        Formatted markdown string
    """
    output = []
    output.append("# Training Metrics Comparison\n")
    
    models = list(metrics_dict.keys())
    
    # Overall metrics table
    output.append("## Overall Training Metrics\n")
    output.append("| Metric | " + " | ".join(models) + " |")
    output.append("|" + "---|" * (len(models) + 1))
    
    metric_labels = [
        ('total_epochs', 'Total Epochs'),
        ('best_val_acc', 'Best Validation Acc (%)'),
        ('best_epoch', 'Best Epoch'),
        ('final_train_acc', 'Final Train Acc (%)'),
        ('final_val_acc', 'Final Val Acc (%)'),
        ('train_val_gap', 'Train-Val Gap (%)'),
        ('overall_test_acc', '**Overall Test Acc (%)**'),
        ('final_train_loss', 'Final Train Loss'),
        ('final_val_loss', 'Final Val Loss'),
        ('training_time', 'Training Time (min)'),
        ('status', 'Overfitting Status'),
    ]
    
    for key, label in metric_labels:
        values = []
        for model in models:
            if key in metrics_dict[model]:
                val = metrics_dict[model][key]
                if isinstance(val, float):
                    if 'loss' in key.lower():
                        values.append(f"{val:.4f}")
                    elif 'time' in key.lower():
                        values.append(f"{val:.2f}")
                    else:
                        values.append(f"{val:.2f}")
                else:
                    values.append(str(val))
            else:
                values.append("N/A")
        
        output.append(f"| {label} | " + " | ".join(values) + " |")
    
    output.append("\n")
    
    # Per-class accuracy
    output.append("## Per-Class Test Accuracy\n")
    
    all_classes = set()
    for model in models:
        if 'per_class_test' in metrics_dict[model]:
            all_classes.update(metrics_dict[model]['per_class_test'].keys())
    
    if all_classes:
        output.append("| Class | " + " | ".join(models) + " |")
        output.append("|" + "---|" * (len(models) + 1))
        
        for class_name in sorted(all_classes):
            values = []
            for model in models:
                if 'per_class_test' in metrics_dict[model] and class_name in metrics_dict[model]['per_class_test']:
                    acc = metrics_dict[model]['per_class_test'][class_name]['accuracy']
                    values.append(f"{acc:.1f}%")
                else:
                    values.append("N/A")
            
            output.append(f"| {class_name} | " + " | ".join(values) + " |")
    
    output.append("\n")
    
    # Summary
    output.append("## Summary Analysis\n")
    
    test_accs = {model: metrics_dict[model].get('overall_test_acc', 0) for model in models}
    best_accuracy_model = max(test_accs, key=test_accs.get)
    
    train_times = {model: metrics_dict[model].get('training_time', float('inf')) for model in models}
    fastest_model = min(train_times, key=train_times.get)
    
    gaps = {model: metrics_dict[model].get('train_val_gap', float('inf')) for model in models}
    least_overfit_model = min(gaps, key=gaps.get)
    
    output.append(f"- 🏆 **Best Test Accuracy:** {best_accuracy_model} ({test_accs[best_accuracy_model]:.2f}%)")
    output.append(f"- ⚡ **Fastest Training:** {fastest_model} ({train_times[fastest_model]:.2f} min)")
    output.append(f"- ✅ **Least Overfitting:** {least_overfit_model} (gap: {gaps[least_overfit_model]:.2f}%)\n")
    
    # Rankings
    output.append("### Accuracy Ranking\n")
    sorted_by_acc = sorted(test_accs.items(), key=lambda x: x[1], reverse=True)
    for rank, (model, acc) in enumerate(sorted_by_acc, 1):
        output.append(f"{rank}. **{model}**: {acc:.2f}%")
    
    output.append("\n### Training Speed Ranking\n")
    sorted_by_speed = sorted(train_times.items(), key=lambda x: x[1])
    for rank, (model, time) in enumerate(sorted_by_speed, 1):
        if time != float('inf'):
            output.append(f"{rank}. **{model}**: {time:.2f} min")
    
    return '\n'.join(output)


def main():
    parser = argparse.ArgumentParser(description='Compare training metrics across models')
    parser.add_argument('--mode', type=str, choices=['transfer', 'feature_extraction', 'both'],
                       default='transfer',
                       help='Training mode to compare: transfer (fine-tuning), feature_extraction, or both (default: transfer)')
    parser.add_argument('--mobilenetv2-summary', type=str, default=None,
                       help='Path to MobileNetV2 training summary (overrides default)')
    parser.add_argument('--mobilenetv3-summary', type=str, default=None,
                       help='Path to MobileNetV3-Small training summary (overrides default)')
    parser.add_argument('--shufflenet-summary', type=str, default=None,
                       help='Path to ShuffleNetV2 training summary (overrides default)')
    parser.add_argument('--save', type=str, default='training_comparison.md',
                       help='Output filename for markdown comparison (default: training_comparison.md)')
    parser.add_argument('--save-txt', type=str, default='training_comparison.txt',
                       help='Output filename for text comparison (default: training_comparison.txt)')
    parser.add_argument('--save-json', type=str, default=None,
                       help='Optional: Save raw metrics as JSON')
    
    args = parser.parse_args()
    
    # Get base directory
    base_dir = Path(__file__).parent.parent
    
    # Determine which summaries to compare based on mode
    if args.mode == 'transfer':
        summaries = {
            'MobileNetV2 (Transfer)': args.mobilenetv2_summary or 'models/mobilenetv2/mobilenet_transfer_training_summary.txt',
            'MobileNetV3-Small (Transfer)': args.mobilenetv3_summary or 'models/mobilenetv3-small/mobilenet_v3_small_transfer_training_summary.txt',
            'ShuffleNetV2 (Transfer)': args.shufflenet_summary or 'models/shufflenetv2/shufflenet_v2_transfer_training_summary.txt'
        }
    elif args.mode == 'feature_extraction':
        summaries = {
            'MobileNetV2 (Feature Ext)': args.mobilenetv2_summary or 'models/mobilenetv2/mobilenet_feature_extraction_training_summary.txt',
            'MobileNetV3-Small (Feature Ext)': args.mobilenetv3_summary or 'models/mobilenetv3-small/mobilenetv3_small_feature_extraction_training_summary.txt',
            'ShuffleNetV2 (Feature Ext)': args.shufflenet_summary or 'models/shufflenetv2/shufflenetv2_feature_extraction_training_summary.txt'
        }
    else:  # both
        summaries = {
            'MobileNetV2 (Transfer)': 'models/mobilenetv2/mobilenet_transfer_training_summary.txt',
            'MobileNetV2 (Feature Ext)': 'models/mobilenetv2/mobilenet_feature_extraction_training_summary.txt',
            'MobileNetV3-Small (Transfer)': 'models/mobilenetv3-small/mobilenet_v3_small_transfer_training_summary.txt',
            'MobileNetV3-Small (Feature Ext)': 'models/mobilenetv3-small/mobilenetv3_small_feature_extraction_training_summary.txt',
            'ShuffleNetV2 (Transfer)': 'models/shufflenetv2/shufflenet_v2_transfer_training_summary.txt',
            'ShuffleNetV2 (Feature Ext)': 'models/shufflenetv2/shufflenetv2_feature_extraction_training_summary.txt'
        }
    
    # Convert to Path objects
    summaries = {name: base_dir / path for name, path in summaries.items()}
    
    print("\n" + "="*100)
    print("TRAINING METRICS COMPARISON")
    print("="*100 + "\n")
    
    metrics_dict = {}
    missing_files = []
    
    for model_name, file_path in summaries.items():
        print(f"Reading {model_name}: {file_path}")
        metrics = parse_training_summary(file_path)
        if metrics:
            metrics_dict[model_name] = metrics
            print(f"  ✓ Loaded successfully")
        else:
            print(f"  ✗ File not found")
            missing_files.append(model_name)
    
    print()
    
    if len(metrics_dict) == 0:
        print("❌ Error: No training summary files found!")
        print("\nExpected files:")
        for model_name, file_path in summaries.items():
            print(f"  - {file_path}")
        print("\nPlease train the models first to generate summary files.")
        return
    
    if missing_files:
        print(f"⚠️  Warning: Missing summaries for: {', '.join(missing_files)}")
        print("Comparison will only include available models.\n")
    
    # Generate comparison
    text_output = format_comparison_table(metrics_dict)
    markdown_output = format_markdown_table(metrics_dict)
    
    # Print to console
    print(text_output)
    print()
    
    # Save text output
    txt_path = Path(args.save_txt)
    with open(txt_path, 'w') as f:
        f.write(text_output)
    print(f"✓ Text comparison saved to: {txt_path}")
    
    # Save markdown output
    md_path = Path(args.save)
    with open(md_path, 'w') as f:
        f.write(markdown_output)
    print(f"✓ Markdown comparison saved to: {md_path}")
    
    # Save JSON if requested
    if args.save_json:
        json_path = Path(args.save_json)
        with open(json_path, 'w') as f:
            json.dump(metrics_dict, f, indent=2)
        print(f"✓ JSON metrics saved to: {json_path}")
    
    print("\n" + "="*100)
    print("COMPARISON COMPLETE")
    print("="*100 + "\n")


if __name__ == '__main__':
    main()
