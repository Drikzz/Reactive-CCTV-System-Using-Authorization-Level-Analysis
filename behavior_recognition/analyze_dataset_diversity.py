"""
Analyze dataset to understand perspective diversity.
Helps identify which angles/perspectives are underrepresented.
"""

from pathlib import Path
import cv2
import numpy as np
from collections import defaultdict

def analyze_dataset_diversity(dataset_dir):
    """
    Analyze training dataset to understand diversity.
    
    Args:
        dataset_dir: Path to training dataset
    """
    dataset_dir = Path(dataset_dir)
    
    print("\n" + "="*70)
    print("DATASET DIVERSITY ANALYSIS")
    print("="*70)
    
    for class_folder in dataset_dir.iterdir():
        if not class_folder.is_dir():
            continue
            
        print(f"\n📁 Class: {class_folder.name}")
        print("-" * 70)
        
        images = list(class_folder.glob("*.jpg")) + list(class_folder.glob("*.png")) + list(class_folder.glob("*.jpeg"))
        
        if not images:
            print("  No images found!")
            continue
        
        # Analyze image properties
        aspect_ratios = []
        sizes = []
        brightness_levels = []
        
        for img_path in images:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
                
            h, w = img.shape[:2]
            aspect_ratios.append(w / h)
            sizes.append(w * h)
            brightness_levels.append(np.mean(img))
        
        # Statistics
        print(f"  Total images: {len(images)}")
        print(f"\n  Image Dimensions:")
        print(f"    Aspect ratios: {np.mean(aspect_ratios):.2f} ± {np.std(aspect_ratios):.2f}")
        print(f"    Sizes: {np.mean(sizes)/1000:.0f}K ± {np.std(sizes)/1000:.0f}K pixels")
        print(f"\n  Brightness:")
        print(f"    Average: {np.mean(brightness_levels):.1f} ± {np.std(brightness_levels):.1f}")
        print(f"    Range: {np.min(brightness_levels):.1f} to {np.max(brightness_levels):.1f}")
        
        # Diversity score (higher = more diverse)
        diversity_score = (
            np.std(aspect_ratios) * 100 +  # Aspect ratio variety
            np.std(brightness_levels) / 10   # Lighting variety
        )
        
        print(f"\n  📊 Diversity Score: {diversity_score:.1f}")
        if diversity_score < 15:
            print("     ⚠️  LOW diversity - images are very similar")
            print("     → Consider adding more varied angles/lighting")
        elif diversity_score < 30:
            print("     ✓ MEDIUM diversity - reasonable variation")
        else:
            print("     ✓✓ HIGH diversity - good variation")
        
        # Show sample filenames to help identify patterns
        print(f"\n  📝 Sample filenames:")
        for img_path in sorted(images)[:5]:
            print(f"     {img_path.name}")
        if len(images) > 5:
            print(f"     ... and {len(images) - 5} more")
    
    print("\n" + "="*70)
    print("\n💡 RECOMMENDATIONS:")
    print("="*70)
    print("""
1. Check if images are too similar (low diversity score)
2. Look for missing perspectives in filenames
3. Add 10-15 images of underrepresented angles:
   - Front view (person's face visible)
   - Top-down view
   - Far-away shots
   - Different lighting conditions
   
4. Current issues often come from:
   ❌ All images from same camera angle
   ❌ All images in same lighting
   ❌ All images showing same equipment setup
   
5. Good datasets have:
   ✓ Multiple camera angles
   ✓ Different distances (close-up, medium, far)
   ✓ Various lighting (bright, normal, dim)
   ✓ Different equipment (various laptops, monitors)
    """)


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Analyze dataset diversity')
    parser.add_argument('--data', type=str, required=True,
                       help='Path to training dataset directory')
    
    args = parser.parse_args()
    
    analyze_dataset_diversity(args.data)
