"""
Dlib CNN Face Recognizer with GPU Acceleration
Handles face detection and recognition using GPU-accelerated methods
"""
import os
import cv2
import dlib
import numpy as np
import joblib
import torch
import torchvision.transforms as transforms
from sklearn.svm import SVC
from sklearn.preprocessing import LabelEncoder

class DlibCNNRecognizer:
    def __init__(self, models_dir):
        self.models_dir = models_dir
        # GPU setup
        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        print(f"[INFO] Using device: {self.device}")
        
        # CPU detectors (fallbacks)
        self.detector = None
        self.hog_detector = None
        self.opencv_detector = None
        
        # GPU-accelerated detectors
        self.mtcnn_detector = None
        self.retinaface_detector = None
        self.yolo_face_detector = None
        
        # Dlib components
        self.predictor = None
        self.face_rec_model = None
        self.classifier = None
        self.label_encoder = None
        
        # GPU transforms
        self.transform = transforms.Compose([
            transforms.ToPILImage(),
            transforms.Resize((112, 112)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.5, 0.5, 0.5], std=[0.5, 0.5, 0.5])
        ])
        
        self.loaded = False
        self.load_models()
    
    def load_models(self):
        """Load models with GPU acceleration where possible"""
        try:
            # Load GPU-accelerated face detectors first
            self.load_gpu_detectors()
            
            # Load Dlib models (CPU fallbacks)
            self.load_dlib_models()
            
            # Load classifier
            self.load_classifier()
            
            self.loaded = True
            print("[INFO] All models loaded successfully")
            return True
            
        except Exception as e:
            print(f"[ERROR] Failed to load models: {e}")
            import traceback
            traceback.print_exc()
            return False
    
    def load_gpu_detectors(self):
        """Load GPU-accelerated face detectors"""
        
        # 1. MTCNN with GPU
        try:
            from mtcnn import MTCNN
            self.mtcnn_detector = MTCNN(device=self.device)
            print(f"[INFO] MTCNN detector loaded on {self.device}")
        except Exception as e:
            print(f"[WARN] MTCNN loading failed: {e}")
        
        # 2. RetinaFace with GPU (better than Dlib CNN)
        try:
            from retinaface import RetinaFace
            # RetinaFace automatically uses GPU if available
            self.retinaface_detector = RetinaFace
            print(f"[INFO] RetinaFace detector loaded (GPU-accelerated)")
        except ImportError:
            try:
                # Alternative: insightface RetinaFace
                import insightface
                from insightface.app import FaceAnalysis
                self.retinaface_detector = FaceAnalysis(providers=['CUDAExecutionProvider', 'CPUExecutionProvider'])
                self.retinaface_detector.prepare(ctx_id=0 if torch.cuda.is_available() else -1, det_size=(640, 640))
                print(f"[INFO] InsightFace RetinaFace loaded on GPU")
            except Exception as e:
                print(f"[WARN] RetinaFace loading failed: {e}")
        
        # 3. YOLO-Face (Ultra-fast GPU face detection)
        try:
            from ultralytics import YOLO
            # Try to load YOLO face model
            yolo_face_path = os.path.join("models", "yolov8n-face.pt")
            os.makedirs("models", exist_ok=True)
            
            if not os.path.exists(yolo_face_path):
                # Download YOLO face model
                print("[INFO] Downloading YOLO face model...")
                try:
                    import urllib.request
                    urllib.request.urlretrieve(
                        "https://github.com/akanametov/yolo-face/releases/download/v0.0.0/yolov8n-face.pt",
                        yolo_face_path
                    )
                    print("[INFO] YOLO face model downloaded successfully")
                except Exception as download_error:
                    print(f"[WARN] Failed to download YOLO face model: {download_error}")
                    print("[INFO] Will use other GPU detectors instead")
                    return
            
            self.yolo_face_detector = YOLO(yolo_face_path)
            print(f"[INFO] YOLO-Face detector loaded on GPU")
        except Exception as e:
            print(f"[WARN] YOLO-Face loading failed: {e}")
    
    def load_dlib_models(self):
        """Load Dlib models (CPU fallbacks)"""
        # Dlib model paths
        detector_path = os.path.join(self.models_dir, "mmod_human_face_detector.dat")
        predictor_path = os.path.join(self.models_dir, "shape_predictor_68_face_landmarks.dat")
        face_rec_path = os.path.join(self.models_dir, "dlib_face_recognition_resnet_model_v1.dat")
        
        # Load Dlib models
        if os.path.exists(detector_path):
            print("[INFO] Loading Dlib CNN face detector (CPU fallback)...")
            self.detector = dlib.cnn_face_detection_model_v1(detector_path)
            
        if os.path.exists(predictor_path):
            print("[INFO] Loading shape predictor...")
            self.predictor = dlib.shape_predictor(predictor_path)
            
        if os.path.exists(face_rec_path):
            print("[INFO] Loading face recognition model...")
            self.face_rec_model = dlib.face_recognition_model_v1(face_rec_path)
        
        # Load HOG detector
        self.hog_detector = dlib.get_frontal_face_detector()
        
        # Load OpenCV detectors
        try:
            self.opencv_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
            self.opencv_profile_detector = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_profileface.xml')
        except Exception as e:
            print(f"[WARN] OpenCV detectors failed: {e}")
    
    def load_classifier(self):
        """Load classifier (CPU SVM is fast enough)"""
        classifier_path = os.path.join(self.models_dir, "dlib_svm.joblib")
        encoder_path = os.path.join(self.models_dir, "label_encoder.joblib")
        
        if os.path.exists(classifier_path) and os.path.exists(encoder_path):
            self.classifier = joblib.load(classifier_path)
            self.label_encoder = joblib.load(encoder_path)
            
            print(f"[INFO] Loaded classifier with classes: {list(self.label_encoder.classes_)}")
            print("[INFO] Using CPU SVM classifier (optimized and fast)")
        else:
            print("[WARN] No trained classifier found")
    
    def detect_faces_gpu(self, image):
        """GPU-accelerated face detection"""
        faces = []
        
        # Convert to RGB
        if len(image.shape) == 3 and image.shape[2] == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Try GPU detectors in order of preference
        
        # 1. YOLO-Face (fastest, most accurate)
        if self.yolo_face_detector is not None:
            try:
                results = self.yolo_face_detector(rgb_image, verbose=False)
                for result in results:
                    boxes = result.boxes
                    if boxes is not None:
                        for box in boxes:
                            conf = float(box.conf[0])
                            if conf > 0.5:
                                x1, y1, x2, y2 = map(int, box.xyxy[0].cpu().numpy())
                                rect = dlib.rectangle(x1, y1, x2, y2)
                                faces.append({
                                    'box': (x1, y1, x2-x1, y2-y1),
                                    'confidence': conf,
                                    'rect': rect
                                })
                if faces:
                    return faces
            except Exception as e:
                print(f"[DEBUG] YOLO-Face failed: {e}")
        
        # 2. RetinaFace
        if self.retinaface_detector is not None:
            try:
                if hasattr(self.retinaface_detector, 'get'):  # insightface version
                    results = self.retinaface_detector.get(rgb_image)
                    for face in results:
                        bbox = face.bbox.astype(int)
                        x1, y1, x2, y2 = bbox
                        rect = dlib.rectangle(x1, y1, x2, y2)
                        faces.append({
                            'box': (x1, y1, x2-x1, y2-y1),
                            'confidence': face.det_score,
                            'rect': rect
                        })
                else:  # retinaface package
                    results = self.retinaface_detector.detect_faces(rgb_image)
                    for key, face in results.items():
                        if face['score'] > 0.5:
                            area = face['facial_area']
                            x1, y1, x2, y2 = area
                            rect = dlib.rectangle(x1, y1, x2, y2)
                            faces.append({
                                'box': (x1, y1, x2-x1, y2-y1),
                                'confidence': face['score'],
                                'rect': rect
                            })
                if faces:
                    return faces
            except Exception as e:
                print(f"[DEBUG] RetinaFace failed: {e}")
        
        # 3. MTCNN
        if self.mtcnn_detector is not None:
            try:
                results = self.mtcnn_detector.detect_faces(rgb_image)
                for detection in results:
                    if detection['confidence'] > 0.5:
                        box = detection['box']
                        x, y, w, h = box
                        rect = dlib.rectangle(x, y, x + w, y + h)
                        faces.append({
                            'box': (x, y, w, h),
                            'confidence': detection['confidence'],
                            'rect': rect
                        })
                if faces:
                    return faces
            except Exception as e:
                print(f"[DEBUG] MTCNN failed: {e}")
        
        return faces
    
    def detect_faces(self, image):
        """Detect faces using GPU-first approach with CPU fallbacks"""
        if not self.loaded:
            return []
        
        try:
            # Try GPU detectors first
            faces = self.detect_faces_gpu(image)
            if faces:
                return faces
            
            # Fallback to CPU detectors
            return self.detect_faces_cpu(image)
            
        except Exception as e:
            print(f"[ERROR] Face detection failed: {e}")
            return []
    
    def detect_faces_cpu(self, image):
        """CPU fallback face detection"""
        faces = []
        
        # Convert to RGB
        if len(image.shape) == 3:
            rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        else:
            rgb_image = image
        
        # Try Dlib CNN detector
        if self.detector is not None:
            detections = self.detector(rgb_image)
            
            for detection in detections:
                rect = detection.rect
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()
                
                faces.append({
                    'box': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': detection.confidence,
                    'rect': rect
                })
            
            if faces:
                return faces
            
        # Try HOG detector
        if self.hog_detector is not None:
            hog_faces = self.hog_detector(rgb_image)
            
            for rect in hog_faces:
                x1, y1 = rect.left(), rect.top()
                x2, y2 = rect.right(), rect.bottom()
                
                faces.append({
                    'box': (x1, y1, x2 - x1, y2 - y1),
                    'confidence': 0.8,
                    'rect': rect
                })
            
            if faces:
                return faces
        
        # Try OpenCV detectors
        if self.opencv_detector is not None:
            gray = cv2.cvtColor(rgb_image, cv2.COLOR_RGB2GRAY) if len(rgb_image.shape) == 3 else rgb_image
            opencv_faces = self.opencv_detector.detectMultiScale(gray, 1.1, 4)
            
            # Also try profile detector
            if len(opencv_faces) == 0 and self.opencv_profile_detector is not None:
                opencv_faces = self.opencv_profile_detector.detectMultiScale(gray, 1.1, 4)
            
            for (x, y, w, h) in opencv_faces:
                rect = dlib.rectangle(x, y, x + w, y + h)
                
                faces.append({
                    'box': (x, y, w, h),
                    'confidence': 0.7,
                    'rect': rect
                })
        
        return faces
    
    def get_face_encoding(self, image, face_rect):
        """Get face encoding using Dlib's face recognition model"""
        if not self.loaded:
            return None
        
        try:
            # Convert to RGB if needed
            if len(image.shape) == 3:
                rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            else:
                rgb_image = image
            
            # Get landmarks
            landmarks = self.predictor(rgb_image, face_rect)
            
            # Get face encoding
            encoding = self.face_rec_model.compute_face_descriptor(rgb_image, landmarks)
            
            return np.array(encoding)
            
        except Exception as e:
            print(f"[ERROR] Face encoding failed: {e}")
            return None
    
    def recognize_face_in_crop(self, person_crop, original_frame, person_bbox):
        """Recognize face within person crop (main interface method)"""
        if person_crop is None or person_crop.size == 0:
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
        
        # Quick size check - skip very small crops
        h, w = person_crop.shape[:2]
        if h < 60 or w < 40:  # Too small for reliable face detection
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
        
        # Resize person crop if too large (optimization)
        if h > 300 or w > 200:
            scale = min(300/h, 200/w)
            new_h, new_w = int(h*scale), int(w*scale)
            person_crop = cv2.resize(person_crop, (new_w, new_h))
        
        try:
            # Detect faces in person crop
            faces = self.detect_faces(person_crop)
            
            if not faces:
                return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
            
            # Take the most confident face
            best_face = max(faces, key=lambda f: f['confidence'])
            
            # Get face encoding
            encoding = self.get_face_encoding(person_crop, best_face['rect'])
            
            if encoding is None:
                return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}
            
            # Convert face coordinates to original frame
            px1, py1, px2, py2 = person_bbox
            fx, fy, fw, fh = best_face['box']
            face_bbox_orig = (px1 + fx, py1 + fy, px1 + fx + fw, py1 + fy + fh)
            
            # Classify if classifier is available
            if self.classifier is not None and self.label_encoder is not None:
                try:
                    # Predict
                    probs = self.classifier.predict_proba([encoding])[0]
                    pred_idx = np.argmax(probs)
                    confidence = probs[pred_idx]
                    
                    if confidence >= 0.45:  # Threshold
                        name = self.label_encoder.inverse_transform([pred_idx])[0]
                        return {
                            'name': name,
                            'confidence': confidence,
                            'face_bbox': face_bbox_orig,
                            'face_encoding': encoding  # Return encoding for re-identification
                        }
                    else:
                        # Return encoding even if confidence is low (for re-identification)
                        return {
                            'name': 'Unknown', 
                            'confidence': confidence,
                            'face_bbox': face_bbox_orig,
                            'face_encoding': encoding
                        }
                except Exception as e:
                    print(f"[WARN] Classification failed: {e}")
            
            # Always return face encoding for potential re-identification
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': face_bbox_orig, 'face_encoding': encoding}
            
        except Exception as e:
            print(f"[ERROR] Face recognition in crop failed: {e}")
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None}