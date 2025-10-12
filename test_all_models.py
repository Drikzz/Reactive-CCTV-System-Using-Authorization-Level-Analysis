"""
Comprehensive Face Recognition Model Testing Script

This script tests all three face recognition models (FaceNet, Dlib CNN, ArcFace)
on your dataset and generates detailed performance reports with visualizations.
"""

import os
import sys
import cv2
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import time
import json
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Tuple, Optional
import warnings
warnings.filterwarnings('ignore')

# Add project root to path FIRST
project_root = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, project_root)

# CRITICAL: Patch MTCNN at import time, before ANY model can use it
def patch_mtcnn_globally():
    """Patch MTCNN to handle torch.device arguments"""
    try:
        import torch
        
        # Import and immediately patch MTCNN
        from facenet_pytorch import MTCNN as _OriginalMTCNN
        import facenet_pytorch
        
        class MTCNNDeviceSafe(_OriginalMTCNN):
            def __init__(self, *args, **kwargs):
                # Convert ALL device arguments to strings
                new_args = []
                for arg in args:
                    if isinstance(arg, torch.device):
                        if arg.type == 'cuda':
                            new_args.append(f"cuda:{arg.index or 0}")
                        else:
                            new_args.append(arg.type)
                    else:
                        new_args.append(arg)
                
                new_kwargs = {}
                for key, value in kwargs.items():
                    if key == 'device' and isinstance(value, torch.device):
                        if value.type == 'cuda':
                            new_kwargs[key] = f"cuda:{value.index or 0}"
                        else:
                            new_kwargs[key] = value.type
                    elif hasattr(value, 'type') and hasattr(value, 'index'):
                        # Handle other device-like objects
                        if hasattr(value, 'type') and value.type == 'cuda':
                            new_kwargs[key] = f"cuda:{getattr(value, 'index', 0) or 0}"
                        else:
                            new_kwargs[key] = str(getattr(value, 'type', value))
                    else:
                        new_kwargs[key] = value
                
                # Debug print to see what we're passing
                if 'device' in new_kwargs:
                    print(f"[PATCH] MTCNN device: {new_kwargs['device']} (type: {type(new_kwargs['device'])})")
                
                # Call parent constructor
                super().__init__(*new_args, **new_kwargs)
        
        # Replace MTCNN everywhere possible
        facenet_pytorch.MTCNN = MTCNNDeviceSafe
        
        # Update sys.modules
        import sys
        if 'facenet_pytorch' in sys.modules:
            sys.modules['facenet_pytorch'].MTCNN = MTCNNDeviceSafe
        
        # Patch any existing modules
        for module_name, module in list(sys.modules.items()):
            if module and hasattr(module, 'MTCNN'):
                try:
                    if getattr(module, 'MTCNN', None) is _OriginalMTCNN:
                        setattr(module, 'MTCNN', MTCNNDeviceSafe)
                        print(f"[PATCH] Patched MTCNN in module: {module_name}")
                except Exception:
                    pass
        
        # Also patch the torch.cuda availability check that might cause issues
        original_is_available = torch.cuda.is_available
        def safe_is_available():
            try:
                return original_is_available()
            except Exception:
                return False
        torch.cuda.is_available = safe_is_available
        
        print("[INFO] ✅ MTCNN device safety patch applied globally")
        return True
        
    except Exception as e:
        print(f"[WARN] ❌ Could not patch MTCNN: {e}")
        import traceback
        traceback.print_exc()
        return False

# Apply the patch immediately - BEFORE any other imports
patch_success = patch_mtcnn_globally()

# If patch failed, create a dummy MTCNN that always uses CPU
if not patch_success:
    print("[INFO] Creating fallback MTCNN...")
    try:
        import torch
        
        class MTCNNFallback:
            def __init__(self, *args, **kwargs):
                # Force CPU device
                kwargs['device'] = 'cpu'
                # Try to import and use original MTCNN
                try:
                    from facenet_pytorch import MTCNN as _OrigMTCNN
                    super().__init__(*args, **kwargs)
                except Exception as e:
                    print(f"[ERROR] Fallback MTCNN failed: {e}")
                    self._failed = True
            
            def __call__(self, *args, **kwargs):
                if hasattr(self, '_failed'):
                    return None
                return super().__call__(*args, **kwargs)
        
        # Replace MTCNN with fallback
        import facenet_pytorch
        facenet_pytorch.MTCNN = MTCNNFallback
        
    except Exception as e:
        print(f"[ERROR] Could not create fallback MTCNN: {e}")

# Add all model paths to sys.path before importing
model_paths = [
    os.path.join(project_root, "face_recognition", "Facenet"),
    os.path.join(project_root, "face_recognition", "Dlibs CNN"), 
    os.path.join(project_root, "face_recognition", "ArcFace")
]

for path in model_paths:
    if os.path.exists(path) and path not in sys.path:
        sys.path.insert(0, path)

# Now do model imports after MTCNN is patched
try:
    # Import torch first
    import torch
    
    # FaceNet - should now use the patched MTCNN
    from facenet_pytorch import MTCNN, InceptionResnetV1
    import joblib
    FACENET_AVAILABLE = True
    print("[INFO] ✅ FaceNet imports successful")
except Exception as e:
    print(f"[WARN] ❌ FaceNet not available: {e}")
    FACENET_AVAILABLE = False

# Dlib CNN imports
try:
    dlib_path = os.path.join(project_root, "face_recognition", "Dlibs CNN")
    
    # Check if files exist
    dlib_main_file = os.path.join(dlib_path, "dilib_cnn_main_optimized.py")
    dlib_recognizer_file = os.path.join(dlib_path, "dlib_face_recognizer.py")
    
    if os.path.exists(dlib_main_file) and os.path.exists(dlib_recognizer_file):
        from dilib_cnn_main_optimized import recognize_face_in_crop_optimized
        from dlib_face_recognizer import DlibCNNRecognizer
        DLIB_AVAILABLE = True
        print("[INFO] ✅ Dlib CNN imports successful")
    else:
        raise ImportError(f"Dlib files not found: {dlib_main_file}, {dlib_recognizer_file}")
        
except Exception as e:
    print(f"[WARN] ❌ Dlib CNN not available: {e}")
    DLIB_AVAILABLE = False

try:
    # ArcFace
    from arcface_main import SimpleArcFaceRecognizer
    ARCFACE_AVAILABLE = True
    print("[INFO] ✅ ArcFace imports successful")
except Exception as e:
    print(f"[WARN] ❌ ArcFace not available: {e}")
    ARCFACE_AVAILABLE = False

# Configuration
TEST_DATASET_PATHS = [
    "datasets/faces",  # Main dataset
    "logs/FaceNet/known",  # Captured faces from FaceNet
    "logs/DlibCNN/known",  # Captured faces from Dlib
    "logs/ArcFace/known"   # Captured faces from ArcFace
]
RESULTS_DIR = "model_test_results"
CSV_OUTPUT = os.path.join(RESULTS_DIR, "model_comparison.csv")
PLOTS_DIR = os.path.join(RESULTS_DIR, "plots")

# Test configuration
MIN_IMAGES_PER_PERSON = 3  # Minimum images required per person to include in test
MAX_IMAGES_PER_PERSON = 50  # Maximum images to test per person (for speed)
TEST_IMAGE_SIZE = (160, 160)  # Standard face size
CONFIDENCE_THRESHOLD = 0.5  # Minimum confidence for acceptance

class ModelTester:
    """Comprehensive model testing framework"""
    
    def __init__(self):
        # Use explicit string device names everywhere
        if torch.cuda.is_available():
            self.device_str = "cuda:0"
            self.device = "cuda:0"  # Keep as string to avoid torch.device issues
        else:
            self.device_str = "cpu"
            self.device = "cpu"
        
        self.results = []
        self.models = {}
        self.dataset = {}
        
        # Ensure output directories
        os.makedirs(RESULTS_DIR, exist_ok=True)
        os.makedirs(PLOTS_DIR, exist_ok=True)
        
        print(f"[INFO] Using device: {self.device_str}")
        print(f"[INFO] Results will be saved to: {RESULTS_DIR}")
    
    def load_dataset(self) -> Dict[str, List[str]]:
        """Load and organize test dataset"""
        print(f"\n{'='*50}")
        print("📁 LOADING DATASET")
        print(f"{'='*50}")
        
        dataset = {}
        total_images = 0
        
        for dataset_path in TEST_DATASET_PATHS:
            if not os.path.exists(dataset_path):
                print(f"[SKIP] Dataset path not found: {dataset_path}")
                continue
            
            print(f"[INFO] Scanning: {dataset_path}")
            
            # Look for person folders or images directly
            if os.path.isdir(dataset_path):
                # Method 1: Person folders (datasets/faces/person_name/*.jpg)
                for person_folder in os.listdir(dataset_path):
                    person_path = os.path.join(dataset_path, person_folder)
                    if os.path.isdir(person_path):
                        images = []
                        for img_file in os.listdir(person_path):
                            if img_file.lower().endswith(('.jpg', '.jpeg', '.png')):
                                images.append(os.path.join(person_path, img_file))
                        
                        if len(images) >= MIN_IMAGES_PER_PERSON:
                            # Limit images for speed
                            images = images[:MAX_IMAGES_PER_PERSON]
                            dataset[person_folder] = images
                            total_images += len(images)
                            print(f"  ✅ {person_folder}: {len(images)} images")
                        else:
                            print(f"  ⚠️  {person_folder}: {len(images)} images (too few, skipped)")
                
                # Method 2: Direct images with person names in filename (logs/known/*.jpg)
                direct_images = [f for f in os.listdir(dataset_path) 
                               if f.lower().endswith(('.jpg', '.jpeg', '.png'))]
                
                if direct_images:
                    # Group by person name (assuming format: PersonName_*.jpg)
                    person_groups = {}
                    for img_file in direct_images:
                        # Extract person name from filename
                        parts = img_file.split('_')
                        if len(parts) >= 2:
                            person_name = parts[0]
                            if person_name not in person_groups:
                                person_groups[person_name] = []
                            person_groups[person_name].append(os.path.join(dataset_path, img_file))
                    
                    for person_name, images in person_groups.items():
                        if len(images) >= MIN_IMAGES_PER_PERSON:
                            images = images[:MAX_IMAGES_PER_PERSON]
                            if person_name not in dataset:
                                dataset[person_name] = []
                            dataset[person_name].extend(images)
                            total_images += len(images)
                            print(f"  ✅ {person_name}: {len(images)} images (from filenames)")
        
        print(f"\n📊 Dataset Summary:")
        print(f"  Total persons: {len(dataset)}")
        print(f"  Total images: {total_images}")
        
        if len(dataset) == 0:
            print("❌ No valid dataset found!")
            print("Please ensure you have face images in one of these locations:")
            for path in TEST_DATASET_PATHS:
                print(f"  - {path}")
            return {}
        
        self.dataset = dataset
        return dataset
    
    def load_facenet_model(self) -> Optional[object]:
        """Load FaceNet model"""
        if not FACENET_AVAILABLE:
            return None
        
        try:
            print("[INFO] Loading FaceNet model...")
            
            # Use patched MTCNN with string device
            mtcnn = MTCNN(image_size=160, margin=0, keep_all=False, device=self.device_str)
            
            # For embedder, we need a torch.device, but create it fresh to avoid cached objects
            torch_device = torch.device(self.device_str)
            embedder = InceptionResnetV1(pretrained='vggface2').to(torch_device).eval()
            
            # Load classifier
            models_dir = os.path.join("models", "FaceNet")
            svm_path = os.path.join(models_dir, "facenet_svm.joblib")
            le_path = os.path.join(models_dir, "label_encoder.joblib")
            
            if not os.path.exists(svm_path) or not os.path.exists(le_path):
                print("❌ FaceNet models not found")
                return None
            
            classifier = joblib.load(svm_path)
            label_encoder = joblib.load(le_path)
            
            # Create model wrapper
            class FaceNetWrapper:
                def __init__(self, mtcnn, embedder, classifier, label_encoder, device_str):
                    self.mtcnn = mtcnn
                    self.embedder = embedder
                    self.classifier = classifier
                    self.label_encoder = label_encoder
                    self.device_str = device_str
                
                def recognize(self, image_path):
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            return "Unknown", 0.0
                        
                        rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
                        
                        # Detect face
                        face_img = self.mtcnn(rgb)
                        if face_img is None:
                            return "Unknown", 0.0
                        
                        # Get embedding - create fresh torch.device to avoid issues
                        device = torch.device(self.device_str)
                        face_img = face_img.unsqueeze(0).to(device)
                        with torch.no_grad():
                            embedding = self.embedder(face_img).cpu().numpy()[0]
                        
                        # Classify
                        probs = self.classifier.predict_proba([embedding])[0]
                        pred_idx = np.argmax(probs)
                        confidence = probs[pred_idx]
                        
                        if confidence >= CONFIDENCE_THRESHOLD:
                            name = self.label_encoder.inverse_transform([pred_idx])[0]
                            return name, confidence
                        
                        return "Unknown", confidence
                    except Exception as e:
                        print(f"[DEBUG] FaceNet recognition error: {e}")
                        return "Unknown", 0.0
            
            model = FaceNetWrapper(mtcnn, embedder, classifier, label_encoder, self.device_str)
            print("✅ FaceNet model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load FaceNet: {e}")
            import traceback
            traceback.print_exc()
            return None
    
    def load_dlib_model(self) -> Optional[object]:
        """Load Dlib CNN model"""
        if not DLIB_AVAILABLE:
            return None
        
        try:
            print("[INFO] Loading Dlib CNN model...")
            
            # Simple approach - just initialize normally and handle errors gracefully
            models_dir = os.path.join("models", "Dlib")
            os.makedirs(models_dir, exist_ok=True)
            
            print(f"[DEBUG] Initializing DlibCNNRecognizer with models_dir: {models_dir}")
            recognizer = DlibCNNRecognizer(models_dir)
            print(f"[DEBUG] DlibCNNRecognizer initialized successfully")
            
            # Create model wrapper
            class DlibWrapper:
                def __init__(self, recognizer):
                    self.recognizer = recognizer
                
                def recognize(self, image_path):
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            return "Unknown", 0.0
                        
                        # Try different method names in order of preference
                        h, w = img.shape[:2]
                        person_bbox = (0, 0, w, h)
                        
                        result = None
                        
                        # Method 1: Try the optimized version first
                        if hasattr(self.recognizer, 'recognize_face_in_crop_optimized'):
                            try:
                                result = self.recognizer.recognize_face_in_crop_optimized(img, img, person_bbox)
                            except Exception:
                                pass  # Silently try next method
                        
                        # Method 2: Try the regular version
                        if result is None and hasattr(self.recognizer, 'recognize_face_in_crop'):
                            try:
                                result = self.recognizer.recognize_face_in_crop(img, img, person_bbox)
                            except Exception:
                                pass  # Silently try next method
                        
                        # Method 3: Try simple version  
                        if result is None and hasattr(self.recognizer, 'recognize_face'):
                            try:
                                result = self.recognizer.recognize_face(img)
                            except Exception:
                                pass  # Silently try next method
                        
                        # Method 4: Try using the function directly
                        if result is None:
                            try:
                                result = recognize_face_in_crop_optimized(img, img, person_bbox)
                            except Exception:
                                pass  # All methods failed
                        
                        if result is None:
                            return "Unknown", 0.0
                        
                        # Extract name and confidence from result
                        if isinstance(result, dict):
                            name = result.get('name', 'Unknown')
                            confidence = result.get('confidence', 0.0)
                        elif isinstance(result, tuple) and len(result) >= 2:
                            name, confidence = result[0], result[1]
                        else:
                            name, confidence = 'Unknown', 0.0
                        
                        return name, confidence
                    except Exception as e:
                        # Only print MTCNN errors, suppress others for cleaner output
                        if "MTCNN" in str(e):
                            print(f"[DEBUG] MTCNN failed: {e}")
                        return "Unknown", 0.0
            
            model = DlibWrapper(recognizer)
            print("✅ Dlib CNN model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load Dlib CNN: {e}")
            # Print the actual error to see what's wrong
            print(f"[DEBUG] Dlib error details: {str(e)}")
            return None
    
    def load_arcface_model(self) -> Optional[object]:
        """Load ArcFace model"""
        if not ARCFACE_AVAILABLE:
            return None
        
        try:
            print("[INFO] Loading ArcFace model...")
            
            recognizer = SimpleArcFaceRecognizer()
            if not recognizer.load_models():
                print("❌ ArcFace models not found")
                return None
            
            # Create model wrapper
            class ArcFaceWrapper:
                def __init__(self, recognizer):
                    self.recognizer = recognizer
                
                def recognize(self, image_path):
                    try:
                        img = cv2.imread(image_path)
                        if img is None:
                            return "Unknown", 0.0
                        
                        # ArcFace expects BGR numpy array and returns tuple
                        name, confidence = self.recognizer.recognize_face(img)
                        return name, confidence
                    except Exception as e:
                        print(f"[DEBUG] ArcFace recognition error: {e}")
                        return "Unknown", 0.0
            
            model = ArcFaceWrapper(recognizer)
            print("✅ ArcFace model loaded successfully")
            return model
            
        except Exception as e:
            print(f"❌ Failed to load ArcFace: {e}")
            return None
    
    def load_all_models(self):
        """Load all available models"""
        print(f"\n{'='*50}")
        print("🔧 LOADING MODELS")
        print(f"{'='*50}")
        
        if FACENET_AVAILABLE:
            self.models['FaceNet'] = self.load_facenet_model()
        
        if DLIB_AVAILABLE:
            self.models['Dlib_CNN'] = self.load_dlib_model()
        
        if ARCFACE_AVAILABLE:
            self.models['ArcFace'] = self.load_arcface_model()
        
        loaded_models = [name for name, model in self.models.items() if model is not None]
        print(f"\n📊 Loaded Models: {loaded_models}")
        
        if not loaded_models:
            print("❌ No models could be loaded!")
            return False
        
        return True

    def test_single_image(self, model_name: str, model: object, image_path: str, true_label: str) -> Dict:
        """Test a single image with a model"""
        start_time = time.time()
        
        try:
            predicted_label, confidence = model.recognize(image_path)
            inference_time = time.time() - start_time
            
            # Determine if prediction is correct
            correct = (predicted_label == true_label)
            
            return {
                'model': model_name,
                'image_path': image_path,
                'true_label': true_label,
                'predicted_label': predicted_label,
                'confidence': confidence,
                'correct': correct,
                'inference_time': inference_time,
                'accepted': confidence >= CONFIDENCE_THRESHOLD,
                'timestamp': datetime.now().isoformat()
            }
            
        except Exception as e:
            return {
                'model': model_name,
                'image_path': image_path,
                'true_label': true_label,
                'predicted_label': 'Error',
                'confidence': 0.0,
                'correct': False,
                'inference_time': time.time() - start_time,
                'accepted': False,
                'timestamp': datetime.now().isoformat(),
                'error': str(e)
            }
    
    def run_comprehensive_test(self):
        """Run comprehensive testing on all models and dataset"""
        print(f"\n{'='*50}")
        print("🧪 RUNNING COMPREHENSIVE TESTS")
        print(f"{'='*50}")
        
        if not self.dataset:
            print("❌ No dataset loaded!")
            return
        
        if not self.models:
            print("❌ No models loaded!")
            return
        
        total_tests = len(self.dataset) * sum(len(images) for images in self.dataset.values()) * len(self.models)
        current_test = 0
        
        print(f"Total tests to run: {total_tests}")
        print(f"Confidence threshold: {CONFIDENCE_THRESHOLD}")
        
        for model_name, model in self.models.items():
            if model is None:
                continue
            
            print(f"\n🔬 Testing {model_name}...")
            
            for person_name, image_paths in self.dataset.items():
                print(f"  Testing {person_name} ({len(image_paths)} images)...")
                
                for image_path in image_paths:
                    current_test += 1
                    
                    # Show progress
                    if current_test % 10 == 0:
                        progress = (current_test / total_tests) * 100
                        print(f"    Progress: {progress:.1f}% ({current_test}/{total_tests})")
                    
                    # Test single image
                    result = self.test_single_image(model_name, model, image_path, person_name)
                    self.results.append(result)
        
        print(f"\n✅ Testing completed! {len(self.results)} results collected")
    
    def save_results_to_csv(self):
        """Save results to CSV file"""
        if not self.results:
            print("❌ No results to save!")
            return
        
        print(f"\n💾 Saving results to CSV...")
        
        df = pd.DataFrame(self.results)
        df.to_csv(CSV_OUTPUT, index=False)
        
        print(f"✅ Results saved to: {CSV_OUTPUT}")
        print(f"📊 Total records: {len(df)}")
    
    def generate_summary_stats(self) -> pd.DataFrame:
        """Generate summary statistics"""
        if not self.results:
            return pd.DataFrame()
        
        df = pd.DataFrame(self.results)
        
        # Calculate metrics per model
        summary_stats = []
        
        for model_name in df['model'].unique():
            model_df = df[df['model'] == model_name]
            
            total_predictions = len(model_df)
            correct_predictions = len(model_df[model_df['correct'] == True])
            accepted_predictions = len(model_df[model_df['accepted'] == True])
            
            # Calculate metrics
            accuracy = correct_predictions / total_predictions if total_predictions > 0 else 0
            acceptance_rate = accepted_predictions / total_predictions if total_predictions > 0 else 0
            
            # Precision and Recall for accepted predictions only
            accepted_df = model_df[model_df['accepted'] == True]
            if len(accepted_df) > 0:
                accepted_correct = len(accepted_df[accepted_df['correct'] == True])
                precision = accepted_correct / len(accepted_df)
            else:
                precision = 0
            
            # Average confidence and inference time
            avg_confidence = model_df['confidence'].mean()
            avg_inference_time = model_df['inference_time'].mean()
            
            summary_stats.append({
                'Model': model_name,
                'Total_Tests': total_predictions,
                'Correct_Predictions': correct_predictions,
                'Overall_Accuracy': accuracy,
                'Accepted_Predictions': accepted_predictions,
                'Acceptance_Rate': acceptance_rate,
                'Precision_on_Accepted': precision,
                'Avg_Confidence': avg_confidence,
                'Avg_Inference_Time_ms': avg_inference_time * 1000,
                'Total_Inference_Time_s': model_df['inference_time'].sum()
            })
        
        summary_df = pd.DataFrame(summary_stats)
        
        # Save summary
        summary_path = os.path.join(RESULTS_DIR, "model_summary.csv")
        summary_df.to_csv(summary_path, index=False)
        
        return summary_df
    
    def create_visualizations(self, summary_df: pd.DataFrame):
        """Create comprehensive visualizations"""
        if summary_df.empty:
            return
        
        print(f"\n📊 Creating visualizations...")
        
        # Set style
        plt.style.use('default')
        sns.set_palette("husl")
        
        fig = plt.figure(figsize=(15, 10))
        
        # 1. Accuracy Comparison
        plt.subplot(2, 3, 1)
        bars = plt.bar(summary_df['Model'], summary_df['Overall_Accuracy'])
        plt.title('Overall Accuracy by Model', fontsize=12, fontweight='bold')
        plt.ylabel('Accuracy')
        plt.ylim(0, 1)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        
        # 2. Average Inference Time
        plt.subplot(2, 3, 2)
        bars = plt.bar(summary_df['Model'], summary_df['Avg_Inference_Time_ms'])
        plt.title('Average Inference Time', fontsize=12, fontweight='bold')
        plt.ylabel('Time (ms)')
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.1,
                    f'{height:.1f}ms', ha='center', va='bottom')
        plt.xticks(rotation=45)
        
        # 3. Confidence Distribution
        plt.subplot(2, 3, 3)
        bars = plt.bar(summary_df['Model'], summary_df['Avg_Confidence'])
        plt.title('Average Confidence', fontsize=12, fontweight='bold')
        plt.ylabel('Confidence')
        plt.ylim(0, 1)
        for i, bar in enumerate(bars):
            height = bar.get_height()
            plt.text(bar.get_x() + bar.get_width()/2., height + 0.01,
                    f'{height:.3f}', ha='center', va='bottom')
        plt.xticks(rotation=45)
        
        plt.tight_layout()
        
        # Save plot
        plot_path = os.path.join(PLOTS_DIR, "model_comparison.png")
        plt.savefig(plot_path, dpi=300, bbox_inches='tight')
        print(f"✅ Saved comparison plot: {plot_path}")
        plt.show()
    
    def print_detailed_report(self, summary_df: pd.DataFrame):
        """Print detailed text report"""
        print(f"\n{'='*60}")
        print("📋 DETAILED MODEL COMPARISON REPORT")
        print(f"{'='*60}")
        
        print(f"Test Configuration:")
        print(f"  Confidence Threshold: {CONFIDENCE_THRESHOLD}")
        print(f"  Min Images per Person: {MIN_IMAGES_PER_PERSON}")
        print(f"  Max Images per Person: {MAX_IMAGES_PER_PERSON}")
        print(f"  Total Persons Tested: {len(self.dataset)}")
        print(f"  Total Images Tested: {sum(len(images) for images in self.dataset.values())}")
        
        print(f"\nDataset Composition:")
        for person, images in self.dataset.items():
            print(f"  {person}: {len(images)} images")
        
        print(f"\n{'='*60}")
        print("📊 MODEL PERFORMANCE SUMMARY")
        print(f"{'='*60}")
        
        for _, row in summary_df.iterrows():
            print(f"\n🔬 {row['Model']} Results:")
            print(f"  Total Tests: {row['Total_Tests']}")
            print(f"  Overall Accuracy: {row['Overall_Accuracy']:.3f} ({row['Correct_Predictions']}/{row['Total_Tests']})")
            print(f"  Acceptance Rate: {row['Acceptance_Rate']:.3f} ({row['Accepted_Predictions']}/{row['Total_Tests']})")
            print(f"  Precision on Accepted: {row['Precision_on_Accepted']:.3f}")
            print(f"  Avg Confidence: {row['Avg_Confidence']:.3f}")
            print(f"  Avg Inference Time: {row['Avg_Inference_Time_ms']:.2f}ms")
        
        # Find best performing model
        if len(summary_df) > 1:
            best_accuracy_model = summary_df.loc[summary_df['Overall_Accuracy'].idxmax()]
            best_speed_model = summary_df.loc[summary_df['Avg_Inference_Time_ms'].idxmin()]
            
            print(f"\n{'='*60}")
            print("🏆 BEST PERFORMERS")
            print(f"{'='*60}")
            print(f"🎯 Best Accuracy: {best_accuracy_model['Model']} ({best_accuracy_model['Overall_Accuracy']:.3f})")
            print(f"⚡ Fastest: {best_speed_model['Model']} ({best_speed_model['Avg_Inference_Time_ms']:.2f}ms)")

def main():
    """Main testing function"""
    print("🧪 COMPREHENSIVE FACE RECOGNITION MODEL TESTING")
    print("=" * 60)
    print("This script will test all available models on your dataset")
    print("and generate detailed performance reports with visualizations.")
    print("=" * 60)
    
    # Initialize tester
    tester = ModelTester()
    
    # Load dataset
    dataset = tester.load_dataset()
    if not dataset:
        print("❌ No dataset found. Please ensure you have face images in the expected locations.")
        return
    
    # Load models
    if not tester.load_all_models():
        print("❌ No models could be loaded. Please ensure models are trained and available.")
        return
    
    print("✅ Setup complete! Models and dataset loaded successfully.")
    
    # Calculate total tests
    total_images = sum(len(images) for images in dataset.values())
    total_tests = total_images * len(tester.models)
    
    print(f"\n📊 Test Overview:")
    print(f"  Models to test: {len(tester.models)}")
    print(f"  Total images: {total_images}")
    print(f"  Total tests: {total_tests}")
    
    # Ask user to continue
    try:
        response = input(f"\n⚠️  This will run {total_tests} tests. Continue? (y/N): ").strip().lower()
        if response not in ['y', 'yes']:
            print("Testing cancelled.")
            return
    except KeyboardInterrupt:
        print("\nTesting cancelled.")
        return
    
    # Run comprehensive testing
    start_time = time.time()
    tester.run_comprehensive_test()
    total_time = time.time() - start_time
    
    print(f"\n⏱️  Total testing time: {total_time:.2f} seconds")
    
    # Save results and generate reports
    tester.save_results_to_csv()
    summary_df = tester.generate_summary_stats()
    tester.create_visualizations(summary_df)
    tester.print_detailed_report(summary_df)
    
    print(f"\n✅ TESTING COMPLETED!")
    print(f"📁 All results saved to: {RESULTS_DIR}")

if __name__ == "__main__":
    main()
