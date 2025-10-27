"""
Evaluate FaceNet recognition on YOLOv8-extracted face crops using dataset centroids as ground truth.
This script:
1. Loads face crops extracted by YOLOv8
2. Uses FaceNet model (from facenet_main.py) for recognition
3. Compares predictions against dataset centroids (embeddings.npy/names.npy)
4. Generates accuracy metrics and detailed CSV results
"""

import sys
import os
from pathlib import Path
import csv
import time
from datetime import datetime
from collections import defaultdict, Counter
import numpy as np
import cv2
import torch
import joblib

# Add paths for imports
repo_root = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(repo_root))
sys.path.insert(0, str(repo_root / "face_recognition"))

from facenet_pytorch import InceptionResnetV1, MTCNN, fixed_image_standardization

# -------------------- CONFIG --------------------
CROPS_DIR = repo_root / "face_recognition" / "face-recognition-system" / "data" / "face_crops"
DATASET_ROOT = repo_root / "datasets" / "faces"
MODELS_DIR = repo_root / "models" / "FaceNet"
OUTPUT_DIR = repo_root / "evaluation_results"

# Model paths
SVM_PATH = MODELS_DIR / "facenet_svm.joblib"
LE_PATH = MODELS_DIR / "label_encoder.joblib"
CENTROIDS_PATH = MODELS_DIR / "class_centroids.pkl"
THRESHOLD_PATH = MODELS_DIR / "distance_threshold.npy"

# Recognition parameters (from facenet_main.py)
RECOG_THRESHOLD = 0.45
RECOG_MARGIN = 0.08
FACE_SIZE = 160
DATASET_MATCH_THRESHOLD = 0.85  # Cosine similarity threshold for dataset matching

# Processing
ENABLE_CLAHE = True
CLAHE_CLIP = 2.0
CLAHE_TILE = 8

# -------------------- HELPER FUNCTIONS --------------------

def _norm(s):
    """Normalize string for comparison (alphanumeric lowercase)"""
    if s is None or s == "":
        return ""
    return "".join(c.lower() for c in str(s) if c.isalnum())

def load_dataset_embeddings(dataset_root):
    """Load dataset embeddings and names"""
    emb_path = Path(dataset_root) / "embeddings.npy"
    names_path = Path(dataset_root) / "names.npy"
    
    if not emb_path.exists() or not names_path.exists():
        print(f"[WARN] Dataset embeddings not found at {dataset_root}")
        return None, None
    
    try:
        embeddings = np.load(str(emb_path)).astype(np.float32)
        names = [str(x) for x in np.load(str(names_path)).astype(str)]
        
        # Normalize embeddings
        embeddings = embeddings / (np.linalg.norm(embeddings, axis=1, keepdims=True) + 1e-10)
        
        print(f"[INFO] Loaded {len(names)} dataset embeddings: {names}")
        return embeddings, names
    except Exception as e:
        print(f"[ERROR] Failed to load dataset embeddings: {e}")
        return None, None

def apply_clahe_rgb(rgb_image, clipLimit=CLAHE_CLIP, tileGridSize=CLAHE_TILE):
    """Apply CLAHE preprocessing to RGB image"""
    if rgb_image is None or rgb_image.size == 0:
        return rgb_image
    
    img = rgb_image
    if img.dtype != 'uint8':
        img = (np.clip(img, 0.0, 1.0) * 255).astype('uint8')
    
    lab = cv2.cvtColor(img, cv2.COLOR_RGB2LAB)
    l, a, b = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=clipLimit, tileGridSize=(tileGridSize, tileGridSize))
    cl = clahe.apply(l)
    limg = cv2.merge((cl, a, b))
    return cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)

def preprocess_face(face_bgr):
    """Preprocess face crop for FaceNet (matches facenet_main.py preprocessing)"""
    try:
        # Apply CLAHE if enabled
        if ENABLE_CLAHE:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
            face_rgb = apply_clahe_rgb(face_rgb, CLAHE_CLIP, CLAHE_TILE)
        else:
            face_rgb = cv2.cvtColor(face_bgr, cv2.COLOR_BGR2RGB)
        
        # Resize to 160x160
        face_resized = cv2.resize(face_rgb, (FACE_SIZE, FACE_SIZE))
        
        return face_resized
    except Exception as e:
        print(f"[WARN] Preprocessing failed: {e}")
        return None

def get_facenet_embedding(face_rgb, embedder, device):
    """Get FaceNet embedding for a face (matches facenet_main.py logic)"""
    try:
        # Convert to tensor and standardize
        tensor = torch.from_numpy(face_rgb).permute(2, 0, 1).float()
        tensor = fixed_image_standardization(tensor)
        
        # Get embedding
        with torch.no_grad():
            embedding = embedder(tensor.unsqueeze(0).to(device))
            embedding = embedding.cpu().numpy()[0]
        
        # Normalize
        emb_norm = embedding / (np.linalg.norm(embedding) + 1e-10)
        
        return emb_norm
    except Exception as e:
        print(f"[WARN] Embedding failed: {e}")
        return None

def recognize_with_facenet(face_bgr, classifier, label_encoder, centroids, dist_threshold, embedder, device):
    """
    Recognize face using FaceNet model (matches facenet_main.py recognition logic)
    Returns: (predicted_name, confidence, embedding)
    """
    try:
        # Preprocess
        face_rgb = preprocess_face(face_bgr)
        if face_rgb is None:
            return "Unknown", 0.0, None
        
        # Get embedding
        embedding = get_facenet_embedding(face_rgb, embedder, device)
        if embedding is None:
            return "Unknown", 0.0, None
        
        # Classify
        probs = classifier.predict_proba([embedding])[0]
        pred_idx = np.argmax(probs)
        confidence = probs[pred_idx]
        
        # Check thresholds
        sorted_probs = np.sort(probs)[::-1]
        top2_prob = sorted_probs[1] if len(sorted_probs) > 1 else 0.0
        
        if confidence >= RECOG_THRESHOLD and (confidence - top2_prob) >= RECOG_MARGIN:
            candidate = label_encoder.inverse_transform([pred_idx])[0]
            
            # Distance check if available
            if centroids is not None and dist_threshold is not None:
                centroid = centroids.get(candidate)
                if centroid is not None:
                    dist = float(np.linalg.norm(embedding - centroid))
                    if dist <= dist_threshold:
                        return candidate, confidence, embedding
                    else:
                        return "Unknown", 0.0, embedding
                else:
                    return candidate, confidence, embedding
            else:
                return candidate, confidence, embedding
        
        return "Unknown", 0.0, embedding
        
    except Exception as e:
        print(f"[ERROR] Recognition failed: {e}")
        return "Unknown", 0.0, None

def find_nearest_dataset_match(embedding, dataset_embs, dataset_names, threshold=DATASET_MATCH_THRESHOLD):
    """Find nearest dataset match for an embedding using cosine similarity"""
    if embedding is None or dataset_embs is None:
        return "", 0.0
    
    try:
        # Compute cosine similarities
        sims = dataset_embs.dot(embedding)
        best_idx = int(np.argmax(sims))
        best_score = float(sims[best_idx])
        
        if best_score >= threshold:
            return dataset_names[best_idx], best_score
        else:
            return "", best_score
    except Exception as e:
        print(f"[WARN] Dataset matching failed: {e}")
        return "", 0.0

def process_crops(crops_dir, classifier, label_encoder, centroids, dist_threshold, 
                 embedder, device, dataset_embs, dataset_names):
    """Process all face crops and generate predictions"""
    
    crops_path = Path(crops_dir)
    if not crops_path.exists():
        print(f"[ERROR] Crops directory not found: {crops_dir}")
        return []
    
    crop_files = list(crops_path.glob("*.jpg")) + list(crops_path.glob("*.png"))
    print(f"[INFO] Found {len(crop_files)} crop files")
    
    if len(crop_files) == 0:
        print(f"[WARN] No crop files found in {crops_dir}")
        return []
    
    results = []
    
    for i, crop_file in enumerate(crop_files):
        if i % 50 == 0:
            print(f"[INFO] Processing {i}/{len(crop_files)}...")
        
        try:
            # Load image
            img = cv2.imread(str(crop_file))
            if img is None:
                print(f"[WARN] Failed to load {crop_file.name}")
                continue
            
            start_time = time.time()
            
            # FaceNet recognition
            predicted_name, confidence, embedding = recognize_with_facenet(
                img, classifier, label_encoder, centroids, dist_threshold, embedder, device
            )
            
            inference_time = (time.time() - start_time) * 1000  # ms
            
            # Dataset matching
            dataset_match = ""
            dataset_confidence = 0.0
            if embedding is not None and dataset_embs is not None:
                dataset_match, dataset_confidence = find_nearest_dataset_match(
                    embedding, dataset_embs, dataset_names, DATASET_MATCH_THRESHOLD
                )
            
            # Store result
            result = {
                'file': crop_file.name,
                'predicted': predicted_name,
                'confidence': confidence,
                'inference_time_ms': inference_time,
                'dataset_match': dataset_match,
                'dataset_confidence': dataset_confidence,
                'ground_truth': dataset_match,  # Use dataset match as ground truth
                'embedding': embedding.tolist() if embedding is not None else None
            }
            
            results.append(result)
            
        except Exception as e:
            print(f"[ERROR] Failed to process {crop_file.name}: {e}")
            continue
    
    return results

def calculate_metrics(results):
    """Calculate accuracy metrics"""
    
    total = len(results)
    if total == 0:
        return {}
    
    # Count recognitions (not Unknown)
    recognized = [r for r in results if r['predicted'] != 'Unknown']
    recognition_rate = len(recognized) / total
    
    # Dataset-based accuracy (predicted == dataset_match)
    dataset_verified = [r for r in results if r['dataset_match'] and r['dataset_match'] != '']
    
    if len(dataset_verified) > 0:
        correct_dataset = sum(1 for r in dataset_verified 
                            if _norm(r['predicted']) == _norm(r['dataset_match']))
        dataset_accuracy = correct_dataset / len(dataset_verified)
    else:
        correct_dataset = 0
        dataset_accuracy = 0.0
    
    # Confidence stats (for recognized faces)
    if len(recognized) > 0:
        avg_confidence = np.mean([r['confidence'] for r in recognized])
        avg_inference_time = np.mean([r['inference_time_ms'] for r in recognized])
    else:
        avg_confidence = 0.0
        avg_inference_time = 0.0
    
    # Count predictions by name
    pred_counts = Counter(r['predicted'] for r in results if r['predicted'] != 'Unknown')
    
    metrics = {
        'total_crops': total,
        'recognized': len(recognized),
        'recognition_rate': recognition_rate,
        'dataset_verified': len(dataset_verified),
        'correct_dataset': correct_dataset,
        'dataset_accuracy': dataset_accuracy,
        'avg_confidence': avg_confidence,
        'avg_inference_time_ms': avg_inference_time,
        'predictions': dict(pred_counts.most_common(10))
    }
    
    return metrics

def save_results(results, metrics, output_dir):
    """Save results to CSV and metrics to text file"""
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Save detailed CSV
    csv_path = output_path / f"facenet_evaluation_{timestamp}.csv"
    with open(csv_path, 'w', newline='', encoding='utf-8') as f:
        if results:
            # Get all fieldnames excluding embedding
            fieldnames = [k for k in results[0].keys() if k != 'embedding']
            writer = csv.DictWriter(f, fieldnames=fieldnames)
            writer.writeheader()
            
            for r in results:
                row = {k: v for k, v in r.items() if k != 'embedding'}
                writer.writerow(row)
    
    print(f"[INFO] Results saved to: {csv_path}")
    
    # Save metrics summary
    metrics_path = output_path / f"facenet_metrics_{timestamp}.txt"
    with open(metrics_path, 'w', encoding='utf-8') as f:
        f.write("=" * 60 + "\n")
        f.write("FaceNet Evaluation Metrics\n")
        f.write("=" * 60 + "\n\n")
        
        f.write(f"Total crops processed: {metrics['total_crops']}\n")
        f.write(f"Recognized (not Unknown): {metrics['recognized']} ({metrics['recognition_rate']*100:.1f}%)\n")
        f.write(f"Dataset verified: {metrics['dataset_verified']}\n")
        f.write(f"Correct predictions: {metrics['correct_dataset']}\n")
        f.write(f"Dataset accuracy: {metrics['dataset_accuracy']*100:.1f}%\n")
        f.write(f"Average confidence: {metrics['avg_confidence']:.3f}\n")
        f.write(f"Average inference time: {metrics['avg_inference_time_ms']:.2f}ms\n\n")
        
        f.write("Top predictions:\n")
        for name, count in metrics['predictions'].items():
            f.write(f"  {name}: {count}\n")
    
    print(f"[INFO] Metrics saved to: {metrics_path}")
    
    return csv_path, metrics_path

def print_summary(metrics):
    """Print evaluation summary"""
    print("\n" + "=" * 60)
    print("FACENET EVALUATION SUMMARY")
    print("=" * 60)
    print(f"Total crops: {metrics['total_crops']}")
    print(f"Recognized: {metrics['recognized']} ({metrics['recognition_rate']*100:.1f}%)")
    print(f"Dataset verified: {metrics['dataset_verified']}")
    print(f"Correct: {metrics['correct_dataset']}")
    print(f"Dataset accuracy: {metrics['dataset_accuracy']*100:.1f}%")
    print(f"Avg confidence: {metrics['avg_confidence']:.3f}")
    print(f"Avg inference time: {metrics['avg_inference_time_ms']:.2f}ms")
    print("\nTop predictions:")
    for name, count in list(metrics['predictions'].items())[:5]:
        print(f"  {name}: {count}")
    print("=" * 60 + "\n")

# -------------------- MAIN --------------------
def main():
    print("[INFO] FaceNet Evaluation on YOLOv8 Crops")
    print(f"[INFO] Crops dir: {CROPS_DIR}")
    print(f"[INFO] Dataset: {DATASET_ROOT}")
    print(f"[INFO] Models: {MODELS_DIR}")
    
    # Setup device
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[INFO] Using device: {device}")
    
    # Load dataset embeddings
    print("\n[INFO] Loading dataset embeddings...")
    dataset_embs, dataset_names = load_dataset_embeddings(DATASET_ROOT)
    
    if dataset_embs is None:
        print("[ERROR] Failed to load dataset embeddings. Cannot proceed.")
        return
    
    # Load FaceNet models
    print("\n[INFO] Loading FaceNet models...")
    try:
        classifier = joblib.load(str(SVM_PATH))
        label_encoder = joblib.load(str(LE_PATH))
        print(f"[INFO] Loaded classifier. Classes: {list(label_encoder.classes_)}")
        
        # Load centroids if available
        centroids = None
        if CENTROIDS_PATH.exists():
            centroids = joblib.load(str(CENTROIDS_PATH))
            # Normalize centroids
            for k, v in list(centroids.items()):
                arr = np.asarray(v, dtype=np.float32)
                centroids[k] = arr / (np.linalg.norm(arr) + 1e-10)
            print(f"[INFO] Loaded {len(centroids)} centroids")
        
        # Load distance threshold if available
        dist_threshold = None
        if THRESHOLD_PATH.exists():
            dist_threshold = float(np.load(str(THRESHOLD_PATH)))
            print(f"[INFO] Loaded distance threshold: {dist_threshold:.3f}")
        
    except Exception as e:
        print(f"[ERROR] Failed to load models: {e}")
        return
    
    # Load embedder
    print("[INFO] Loading FaceNet embedder...")
    embedder = InceptionResnetV1(pretrained="vggface2").to(device).eval()
    
    # Process crops
    print("\n[INFO] Processing crops...")
    results = process_crops(
        CROPS_DIR, classifier, label_encoder, centroids, dist_threshold,
        embedder, device, dataset_embs, dataset_names
    )
    
    if not results:
        print("[ERROR] No results generated")
        return
    
    # Calculate metrics
    print("\n[INFO] Calculating metrics...")
    metrics = calculate_metrics(results)
    
    # Save results
    print("\n[INFO] Saving results...")
    save_results(results, metrics, OUTPUT_DIR)
    
    # Print summary
    print_summary(metrics)

if __name__ == "__main__":
    main()