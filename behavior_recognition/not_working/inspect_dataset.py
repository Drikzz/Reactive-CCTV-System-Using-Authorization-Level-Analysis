"""Script to inspect and visualize the training dataset.

Helps identify issues like:
- Mislabeled data
- Poor quality images
- Class imbalance
- Data leakage
"""
import os
import random
from collections import Counter
import matplotlib.pyplot as plt
from PIL import Image


def count_samples_per_class(root_dir: str, split: str = 'train'):
    """Count samples in each class."""
    split_dir = os.path.join(root_dir, split)
    
    if not os.path.isdir(split_dir):
        print(f"Directory not found: {split_dir}")
        return {}
    
    classes = sorted([d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))])
    
    counts = {}
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    for cls in classes:
        class_dir = os.path.join(split_dir, cls)
        count = sum(1 for f in os.listdir(class_dir) 
                   if os.path.splitext(f)[1].lower() in valid_extensions)
        counts[cls] = count
    
    return counts


def visualize_random_samples(root_dir: str, split: str = 'train', samples_per_class: int = 5):
    """Visualize random samples from each class."""
    split_dir = os.path.join(root_dir, split)
    
    if not os.path.isdir(split_dir):
        print(f"Directory not found: {split_dir}")
        return
    
    classes = sorted([d for d in os.listdir(split_dir) 
                     if os.path.isdir(os.path.join(split_dir, d))])
    
    valid_extensions = {'.jpg', '.jpeg', '.png', '.bmp', '.gif'}
    
    # Collect samples
    all_samples = {}
    for cls in classes:
        class_dir = os.path.join(split_dir, cls)
        images = [f for f in os.listdir(class_dir) 
                 if os.path.splitext(f)[1].lower() in valid_extensions]
        
        # Random sample
        num_samples = min(samples_per_class, len(images))
        sampled = random.sample(images, num_samples) if num_samples > 0 else []
        all_samples[cls] = [(cls, os.path.join(class_dir, img)) for img in sampled]
    
    # Create visualization
    num_classes = len(classes)
    fig, axes = plt.subplots(num_classes, samples_per_class, 
                            figsize=(samples_per_class * 3, num_classes * 3))
    
    if num_classes == 1:
        axes = [axes]
    
    for i, cls in enumerate(classes):
        samples = all_samples[cls]
        for j in range(samples_per_class):
            ax = axes[i][j] if samples_per_class > 1 else axes[i]
            
            if j < len(samples):
                _, img_path = samples[j]
                try:
                    img = Image.open(img_path).convert('RGB')
                    ax.imshow(img)
                    ax.set_title(f"{cls}\n{os.path.basename(img_path)}", fontsize=8)
                except Exception as e:
                    ax.text(0.5, 0.5, f"Error: {e}", 
                           ha='center', va='center', fontsize=8)
            else:
                ax.text(0.5, 0.5, "No image", ha='center', va='center')
            
            ax.axis('off')
    
    plt.suptitle(f'Random Samples from {split.upper()} Set', fontsize=16, fontweight='bold')
    plt.tight_layout()
    
    save_path = f'outputs/dataset_visualization_{split}.png'
    os.makedirs('outputs', exist_ok=True)
    plt.savefig(save_path, dpi=150, bbox_inches='tight')
    print(f"Visualization saved to: {save_path}")
    plt.show()


def analyze_dataset(root_dir: str = 'datasets/behavior_frames'):
    """Comprehensive dataset analysis."""
    print("=" * 70)
    print("DATASET ANALYSIS")
    print("=" * 70)
    
    splits = ['train', 'valid', 'test']
    
    all_counts = {}
    for split in splits:
        counts = count_samples_per_class(root_dir, split)
        all_counts[split] = counts
        
        if counts:
            print(f"\n{split.upper()} Split:")
            print("-" * 40)
            total = sum(counts.values())
            for cls, count in sorted(counts.items()):
                percentage = 100 * count / total if total > 0 else 0
                print(f"  {cls:20s}: {count:5d} samples ({percentage:5.1f}%)")
            print(f"  {'TOTAL':20s}: {total:5d}")
    
    # Check for imbalance
    print("\n" + "=" * 70)
    print("CLASS BALANCE ANALYSIS")
    print("=" * 70)
    
    for split in splits:
        if split in all_counts and all_counts[split]:
            counts = all_counts[split]
            count_values = list(counts.values())
            
            if count_values:
                max_count = max(count_values)
                min_count = min(count_values)
                ratio = max_count / max(min_count, 1)
                
                print(f"\n{split.upper()}:")
                print(f"  Max samples: {max_count}")
                print(f"  Min samples: {min_count}")
                print(f"  Imbalance ratio: {ratio:.2f}:1")
                
                if ratio > 5:
                    print(f"  ⚠ WARNING: Severe imbalance! Consider balancing your dataset.")
                elif ratio > 2:
                    print(f"  ⚠ Moderate imbalance. WeightedRandomSampler should help.")
                else:
                    print(f"  ✓ Relatively balanced.")
    
    # Recommendations
    print("\n" + "=" * 70)
    print("RECOMMENDATIONS")
    print("=" * 70)
    
    train_counts = all_counts.get('train', {})
    if train_counts:
        for cls, count in train_counts.items():
            if count < 100:
                print(f"\n⚠ Class '{cls}' has only {count} training samples.")
                print(f"  Recommendation: Collect at least 200-500 samples per class for better performance.")
    
    # Check if only one class exists
    if len(train_counts) == 1:
        print("\n⚠⚠⚠ CRITICAL: Only ONE class found in training data!")
        print("  The model will always predict this class.")
        print("  You need to add samples for other classes (e.g., 'normal', 'not_stealing', etc.)")


def check_class_overlap():
    """Check for potential issues in class definitions."""
    print("\n" + "=" * 70)
    print("CLASS DEFINITION CHECK")
    print("=" * 70)
    
    print("\nQuestions to consider:")
    print("  1. Is 'stealing' the ONLY class in your dataset?")
    print("  2. Do you have a 'normal' or 'not_stealing' class?")
    print("  3. Are the classes mutually exclusive?")
    print("  4. Could there be label noise (mislabeled samples)?")
    print("\nIf you only have 'stealing' samples:")
    print("  → The model will always predict 'stealing' (it has no alternative)")
    print("  → You need to add 'normal' behavior samples for proper classification")


if __name__ == '__main__':
    import argparse
    
    parser = argparse.ArgumentParser(description='Inspect dataset')
    parser.add_argument('--data_dir', type=str, default='datasets/behavior_frames',
                        help='Dataset directory')
    parser.add_argument('--split', type=str, default='train',
                        help='Split to visualize')
    parser.add_argument('--samples', type=int, default=5,
                        help='Samples per class to visualize')
    
    args = parser.parse_args()
    
    # Analyze dataset
    analyze_dataset(args.data_dir)
    
    # Check class definitions
    check_class_overlap()
    
    # Visualize samples
    print(f"\nGenerating visualization for {args.split} split...")
    visualize_random_samples(args.data_dir, args.split, args.samples)
