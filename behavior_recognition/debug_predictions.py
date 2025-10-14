"""
Debug script to analyze what the model is actually seeing/predicting.
Helps identify why one class isn't being recognized.
"""

import sys
from pathlib import Path
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image

BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


def analyze_predictions(model_path, image_paths, class_names=None):
    """
    Analyze predictions on multiple images to find patterns.
    
    Args:
        model_path (str): Path to trained model
        image_paths (list): List of image paths to analyze
        class_names (list): Class names (optional)
    """
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    
    # Load model
    checkpoint = torch.load(model_path, map_location=device)
    num_classes = checkpoint['num_classes']
    if class_names is None:
        class_names = checkpoint.get('class_names', [f'Class_{i}' for i in range(num_classes)])
    
    model = models.mobilenet_v2(weights=None)
    in_features = model.classifier[1].in_features
    model.classifier[1] = nn.Linear(in_features, num_classes)
    model.load_state_dict(checkpoint['model_state_dict'])
    model = model.to(device)
    model.eval()
    
    # Transform
    transform = transforms.Compose([
        transforms.Resize((224, 224)),
        transforms.ToTensor(),
        transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
    ])
    
    print("\n" + "="*70)
    print("DETAILED PREDICTION ANALYSIS")
    print("="*70)
    
    all_predictions = []
    
    for img_path in image_paths:
        img_path = Path(img_path)
        if not img_path.exists():
            print(f"\n⚠️  Image not found: {img_path}")
            continue
        
        # Get true label from folder name
        true_label = img_path.parent.name
        
        # Load and predict
        image = Image.open(img_path).convert('RGB')
        image_tensor = transform(image).unsqueeze(0).to(device)
        
        with torch.no_grad():
            outputs = model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)[0]
        
        # Get prediction
        predicted_idx = torch.argmax(probabilities).item()
        predicted_class = class_names[predicted_idx]
        confidence = probabilities[predicted_idx].item()
        
        # Store results
        result = {
            'path': str(img_path),
            'true_label': true_label,
            'predicted': predicted_class,
            'confidence': confidence,
            'correct': predicted_class == true_label,
            'probabilities': {class_names[i]: probabilities[i].item() for i in range(num_classes)}
        }
        all_predictions.append(result)
        
        # Print detailed info
        print(f"\n📸 {img_path.name}")
        print(f"   True: {true_label}")
        print(f"   Predicted: {predicted_class} ({confidence*100:.1f}%)")
        
        status = "✅ CORRECT" if result['correct'] else "❌ WRONG"
        print(f"   {status}")
        
        print(f"   Probabilities:")
        for cls, prob in result['probabilities'].items():
            marker = "→" if cls == predicted_class else " "
            print(f"     {marker} {cls}: {prob:.4f} ({prob*100:.1f}%)")
    
    # Summary statistics
    print("\n" + "="*70)
    print("SUMMARY STATISTICS")
    print("="*70)
    
    # Overall accuracy
    correct = sum(1 for p in all_predictions if p['correct'])
    total = len(all_predictions)
    print(f"\nOverall: {correct}/{total} correct ({correct/total*100:.1f}%)")
    
    # Per-class accuracy
    print("\nPer-Class Analysis:")
    for class_name in class_names:
        class_preds = [p for p in all_predictions if p['true_label'] == class_name]
        if class_preds:
            class_correct = sum(1 for p in class_preds if p['correct'])
            class_total = len(class_preds)
            avg_conf = np.mean([p['confidence'] for p in class_preds])
            
            print(f"\n  {class_name}:")
            print(f"    Accuracy: {class_correct}/{class_total} ({class_correct/class_total*100:.1f}%)")
            print(f"    Avg Confidence: {avg_conf:.4f} ({avg_conf*100:.1f}%)")
            
            # Show common misclassifications
            misclassified = [p for p in class_preds if not p['correct']]
            if misclassified:
                print(f"    Misclassified as:")
                for p in misclassified:
                    print(f"      - {p['predicted']} (conf: {p['confidence']:.2f}) - {Path(p['path']).name}")
    
    # Confidence distribution
    print("\n" + "="*70)
    print("CONFIDENCE DISTRIBUTION")
    print("="*70)
    
    low_conf = [p for p in all_predictions if p['confidence'] < 0.6]
    med_conf = [p for p in all_predictions if 0.6 <= p['confidence'] < 0.9]
    high_conf = [p for p in all_predictions if p['confidence'] >= 0.9]
    
    print(f"\nLow confidence (<60%): {len(low_conf)}")
    print(f"Medium confidence (60-90%): {len(med_conf)}")
    print(f"High confidence (>90%): {len(high_conf)}")
    
    if low_conf:
        print(f"\n⚠️  Low confidence predictions:")
        for p in low_conf[:5]:  # Show first 5
            print(f"   {Path(p['path']).name}: {p['predicted']} ({p['confidence']*100:.1f}%)")
    
    print("\n" + "="*70)
    
    return all_predictions


if __name__ == "__main__":
    import argparse
    
    parser = argparse.ArgumentParser(description='Debug model predictions')
    parser.add_argument('--model', type=str, required=True,
                       help='Path to model checkpoint')
    parser.add_argument('--test-dir', type=str, required=True,
                       help='Path to test directory with class subfolders')
    parser.add_argument('--classes', type=str, nargs='+',
                       help='Class names (optional)')
    
    args = parser.parse_args()
    
    # Collect all test images
    test_dir = Path(args.test_dir)
    image_paths = []
    
    for class_folder in test_dir.iterdir():
        if class_folder.is_dir():
            images = list(class_folder.glob('*.jpg')) + list(class_folder.glob('*.png'))
            image_paths.extend(images)
    
    if not image_paths:
        print(f"No images found in {test_dir}")
        sys.exit(1)
    
    print(f"Found {len(image_paths)} test images")
    
    # Analyze
    results = analyze_predictions(
        model_path=args.model,
        image_paths=image_paths,
        class_names=args.classes
    )
