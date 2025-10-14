"""
Visualize the misclassified images to understand why the model is wrong.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import matplotlib.pyplot as plt

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


def visualize_misclassified_images(image_paths, model_path, save_dir='outputs'):
    """
    Display the misclassified images side by side to spot patterns.
    
    Args:
        image_paths (list): Paths to misclassified images
        model_path (str): Path to model checkpoint
        save_dir (str): Where to save visualization
    """
    save_dir = Path(save_dir)
    save_dir.mkdir(exist_ok=True)
    
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    class_names = checkpoint.get('class_names', ['Class_0', 'Class_1'])
    
    print("\n" + "="*70)
    print("VISUALIZING MISCLASSIFIED IMAGES")
    print("="*70)
    
    num_images = len(image_paths)
    fig, axes = plt.subplots(1, num_images, figsize=(6*num_images, 6))
    
    if num_images == 1:
        axes = [axes]
    
    for idx, img_path in enumerate(image_paths):
        img_path = Path(img_path)
        
        # Load image
        img = cv2.imread(str(img_path))
        img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        
        # Get true label
        true_label = img_path.parent.name
        
        # Display
        axes[idx].imshow(img)
        axes[idx].set_title(f'{img_path.name}\nTrue: {true_label}\nPredicted: opening-cabinet', 
                           fontsize=10)
        axes[idx].axis('off')
    
    plt.tight_layout()
    output_path = save_dir / 'misclassified_using_computer.png'
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"\n✅ Saved visualization to: {output_path}")
    
    try:
        plt.show()
    except:
        print("(Running in headless mode - check saved file)")
    
    # Also show them individually with more detail
    for img_path in image_paths:
        img_path = Path(img_path)
        img = cv2.imread(str(img_path))
        
        print(f"\n📸 {img_path.name}")
        print(f"   Resolution: {img.shape[1]}x{img.shape[0]}")
        print(f"   Path: {img_path}")
        
        # Save individual copy
        individual_path = save_dir / f"misclassified_{img_path.name}"
        cv2.imwrite(str(individual_path), img)
        print(f"   Saved to: {individual_path}")


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Visualize misclassified images')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test directory')
    parser.add_argument('--images', type=str, nargs='+', required=True,
                       help='Filenames of misclassified images')
    parser.add_argument('--save-dir', type=str, default='outputs',
                       help='Directory to save visualizations')
    
    args = parser.parse_args()
    
    # Find image paths
    test_dir = Path(args.test_dir)
    image_paths = []
    
    for img_name in args.images:
        # Search in using-computer folder
        img_path = test_dir / 'using-computer' / img_name
        if img_path.exists():
            image_paths.append(img_path)
        else:
            print(f"⚠️  Image not found: {img_path}")
    
    if image_paths:
        visualize_misclassified_images(
            image_paths=image_paths,
            model_path=args.model,
            save_dir=args.save_dir
        )
    else:
        print("No images found!")
