"""
Dlib CNN Training Script
Trains a face classifier using Dlib face encodings from dataset images
"""
import os
import cv2
import dlib
import numpy as np
import joblib
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
from sklearn.metrics import classification_report, accuracy_score
import pickle
from pathlib import Path

# Add project root to path
import sys
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import the recognizer to use its models
sys.path.append(os.path.join(os.path.dirname(__file__)))
from dlib_face_recognizer import DlibCNNRecognizer

# -------------------- CONFIG --------------------
DATASETS_DIR = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\datasets\faces"
MODELS_DIR = os.path.join("models", "Dlib")
OUTPUT_DIR = MODELS_DIR

# Training parameters
TEST_SIZE = 0.2
RANDOM_STATE = 42
SVM_KERNEL = 'rbf'
SVM_C = 1.0
SVM_GAMMA = 'scale'

class DlibCNNTrainer:
    def __init__(self):
        self.recognizer = DlibCNNRecognizer(MODELS_DIR)
        self.encodings = []
        self.labels = []
        self.label_encoder = LabelEncoder()
        
    def load_dataset(self, datasets_path):
        """Load images from dataset directory structure"""
        print(f"[INFO] Loading dataset from: {datasets_path}")
        
        if not os.path.exists(datasets_path):
            print(f"[ERROR] Dataset directory not found: {datasets_path}")
            return False
        
        # Get all person directories
        person_dirs = [d for d in os.listdir(datasets_path) 
                      if os.path.isdir(os.path.join(datasets_path, d))]
        
        if not person_dirs:
            print(f"[ERROR] No person directories found in: {datasets_path}")
            return False
        
        print(f"[INFO] Found {len(person_dirs)} persons: {person_dirs}")
        
        total_images = 0
        for person_name in person_dirs:
            person_dir = os.path.join(datasets_path, person_name)
            
            # Get all image files
            image_extensions = ['.jpg', '.jpeg', '.png', '.bmp']
            image_files = []
            for ext in image_extensions:
                image_files.extend(Path(person_dir).glob(f"*{ext}"))
                image_files.extend(Path(person_dir).glob(f"*{ext.upper()}"))
            
            person_encodings = 0
            print(f"[INFO] Processing {person_name}: {len(image_files)} images")
            
            for image_path in image_files:
                try:
                    # Load image
                    image = cv2.imread(str(image_path))
                    if image is None:
                        print(f"[WARN] Could not load image: {image_path}")
                        continue
                    
                    # Get face encoding using Dlib CNN
                    encoding = self.extract_face_encoding(image)
                    
                    if encoding is not None:
                        self.encodings.append(encoding)
                        self.labels.append(person_name)
                        person_encodings += 1
                        total_images += 1
                    else:
                        print(f"[WARN] No face found in: {image_path}")
                        
                except Exception as e:
                    print(f"[ERROR] Processing {image_path}: {e}")
                    continue
            
            print(f"[INFO] {person_name}: {person_encodings} face encodings extracted")
        
        print(f"[INFO] Total encodings extracted: {total_images}")
        return total_images > 0
    
    def extract_face_encoding(self, image):
        """Extract face encoding from image using Dlib CNN"""
        try:
            # Detect faces
            faces = self.recognizer.detect_faces(image)
            
            if not faces:
                return None
            
            # Use the most confident face
            best_face = max(faces, key=lambda f: f['confidence'])
            
            # Get face encoding
            encoding = self.recognizer.get_face_encoding(image, best_face['rect'])
            
            return encoding
            
        except Exception as e:
            print(f"[ERROR] Face encoding failed: {e}")
            return None
    
    def train_classifier(self):
        """Train SVM classifier on face encodings"""
        if len(self.encodings) == 0:
            print("[ERROR] No face encodings found. Cannot train classifier.")
            return False
        
        print(f"[INFO] Training classifier with {len(self.encodings)} encodings...")
        
        # Convert to numpy arrays
        X = np.array(self.encodings)
        y = np.array(self.labels)
        
        # Encode labels
        y_encoded = self.label_encoder.fit_transform(y)
        
        print(f"[INFO] Classes: {list(self.label_encoder.classes_)}")
        print(f"[INFO] Encoding shape: {X.shape}")
        
        # Split dataset
        X_train, X_test, y_train, y_test = train_test_split(
            X, y_encoded, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=y_encoded
        )
        
        print(f"[INFO] Training set: {len(X_train)} samples")
        print(f"[INFO] Test set: {len(X_test)} samples")
        
        # Train SVM classifier
        print("[INFO] Training SVM classifier...")
        classifier = SVC(
            kernel=SVM_KERNEL,
            C=SVM_C,
            gamma=SVM_GAMMA,
            probability=True,  # Enable probability estimates
            random_state=RANDOM_STATE
        )
        
        classifier.fit(X_train, y_train)
        
        # Evaluate on test set
        y_pred = classifier.predict(X_test)
        accuracy = accuracy_score(y_test, y_pred)
        
        print(f"\n[INFO] Training completed!")
        print(f"[INFO] Test Accuracy: {accuracy:.4f}")
        
        # Detailed classification report
        target_names = [self.label_encoder.classes_[i] for i in sorted(set(y_test))]
        print("\n[INFO] Classification Report:")
        print(classification_report(y_test, y_pred, target_names=target_names))
        
        # Save models
        self.save_models(classifier)
        
        return True
    
    def save_models(self, classifier):
        """Save trained classifier and label encoder"""
        os.makedirs(OUTPUT_DIR, exist_ok=True)
        
        # Save classifier
        classifier_path = os.path.join(OUTPUT_DIR, "dlib_svm.joblib")
        joblib.dump(classifier, classifier_path)
        print(f"[INFO] Classifier saved to: {classifier_path}")
        
        # Save label encoder
        encoder_path = os.path.join(OUTPUT_DIR, "label_encoder.joblib")
        joblib.dump(self.label_encoder, encoder_path)
        print(f"[INFO] Label encoder saved to: {encoder_path}")
        
        # Save encodings for debugging/analysis
        encodings_path = os.path.join(OUTPUT_DIR, "face_encodings.pkl")
        with open(encodings_path, 'wb') as f:
            pickle.dump({
                'encodings': self.encodings,
                'labels': self.labels,
                'label_encoder': self.label_encoder
            }, f)
        print(f"[INFO] Face encodings saved to: {encodings_path}")
        
        print(f"\n[INFO] Training complete! Models saved in: {OUTPUT_DIR}")
        print(f"[INFO] You can now run the recognition system.")

def main():
    print("="*60)
    print("Dlib CNN Face Recognition Training")
    print("="*60)
    
    # Check if dataset directory exists
    if not os.path.exists(DATASETS_DIR):
        print(f"[ERROR] Dataset directory not found: {DATASETS_DIR}")
        print("[INFO] Please ensure your dataset is organized as:")
        print("  datasets/")
        print("    person1/")
        print("      image1.jpg")
        print("      image2.jpg")
        print("    person2/")
        print("      image1.jpg")
        print("      ...")
        return
    
    # Initialize trainer
    trainer = DlibCNNTrainer()
    
    # Check if Dlib models are loaded
    if not trainer.recognizer.loaded:
        print("[ERROR] Dlib models not loaded. Please ensure models are downloaded.")
        return
    
    # Load dataset
    if not trainer.load_dataset(DATASETS_DIR):
        print("[ERROR] Failed to load dataset.")
        return
    
    # Train classifier
    if not trainer.train_classifier():
        print("[ERROR] Training failed.")
        return
    
    print("\n" + "="*60)
    print("Training completed successfully!")
    print("You can now run: python \"face_recognition\\Dlibs CNN\\dilib_cnn_main.py\"")
    print("="*60)

if __name__ == "__main__":
    main()