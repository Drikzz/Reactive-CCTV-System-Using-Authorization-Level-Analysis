"""
Integrated Face Recognition + Behavior Recognition System

Combines:
- FaceNet face recognition with authorization levels
- MobileNetV3-Small behavior classification
- YOLOv8 person detection and tracking (ByteTrack)
- Conditional alerts based on authorization levels

Authorization Levels:
- AUTHORIZED: Face tracking only, no behavior monitoring, never alerts
- PARTIAL: Face + behavior tracking, alerts only if no authorized person present
- UNAUTHORIZED: Full tracking, immediate alerts on suspicious behavior

Usage:
    # Run with video file
    python main.py --video path/to/video.mp4
    
    # Run with webcam
    python main.py --webcam
    
    # Save outputs
    python main.py --video path/to/video.mp4 --save outputs/result.mp4 --save-logs
"""

import os
import sys
import cv2
import torch
import numpy as np
import argparse
import time
import joblib
from pathlib import Path
from datetime import datetime
from collections import defaultdict, deque
from PIL import Image

# Deep learning models
from facenet_pytorch import MTCNN, InceptionResnetV1
from torchvision import transforms, models
from torch import nn
from ultralytics import YOLO

# Import authorization configuration
from authorization_config_main import (
    get_authorization_level, should_monitor_behavior, is_suspicious_behavior,
    should_trigger_alert, format_display_name, get_level_color, get_level_thickness,
    get_log_directory, get_annotated_directory, AUTHORIZED, PARTIAL, UNAUTHORIZED,
    SUSPICIOUS_BEHAVIORS, BEHAVIOR_CONFIDENCE_THRESHOLD, print_authorization_summary
)


class IntegratedRecognitionSystem:
    """
    Integrated Face Recognition + Behavior Recognition System
    """
    
    def __init__(self, 
                 yolo_model_path='models/YOLOv8/yolov8m.pt',
                 facenet_model_path='models/FaceNet/inception_resnet_v1.pt',
                 behavior_model_path='models/mobilenetv3-small/mobilenetv3_small_feature_extraction.pth',
                 device=None,
                 save_logs=False,
                 min_class_conf: float = None):
        """
        Initialize integrated system
        
        Args:
            yolo_model_path: Path to YOLOv8 model
            facenet_model_path: Path to FaceNet model
            behavior_model_path: Path to MobileNetV3-Small behavior model
            device: Device to use ('cuda' or 'cpu')
            save_logs: Whether to save logs and frames
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.save_logs = save_logs
        
        print("\n" + "="*80)
        print("INITIALIZING INTEGRATED RECOGNITION SYSTEM")
        print("="*80)
        print(f"Device: {self.device}")
        print(f"Save Logs: {self.save_logs}")
        
        # Initialize YOLO detector
        print(f"\nLoading YOLOv8 from: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        
        # Initialize FaceNet components
        print(f"Loading FaceNet components...")
        self._init_facenet(facenet_model_path)
        
        # Initialize Behavior Recognition
        print(f"Loading Behavior model from: {behavior_model_path}")
        self._init_behavior_model(behavior_model_path)
        
        # Tracking state
        self.track_identities = {}  # {track_id: identity_info}
        self.track_behaviors = {}   # {track_id: behavior_info}
        
        # Alert tracking
        self.alerts_sent = []
        self.alert_cooldown = {}  # {track_id: last_alert_time}
        self.alert_cooldown_seconds = 5.0
        
        # Minimum behavior classification confidence (can be overridden by CLI)
        # If not provided, fall back to config value
        if min_class_conf is not None:
            try:
                self.min_class_conf = float(min_class_conf)
            except Exception:
                self.min_class_conf = BEHAVIOR_CONFIDENCE_THRESHOLD
        else:
            self.min_class_conf = BEHAVIOR_CONFIDENCE_THRESHOLD

        # CSV logging
        self.csv_path = None
        if self.save_logs:
            self._setup_logging()
        
        print("\n✓ System initialized successfully!")
        print("="*80 + "\n")
    
    def _init_facenet(self, model_path):
        """Initialize FaceNet face recognition components"""
        # MTCNN face detector
        self.mtcnn = MTCNN(
            image_size=160,
            margin=0,
            keep_all=True,
            device=self.device
        )
        
        # FaceNet embedder
        self.embedder = InceptionResnetV1(pretrained='vggface2').to(self.device).eval()
        
        # Load classifier
        models_dir = "models/FaceNet"
        svm_path = os.path.join(models_dir, "facenet_svm.joblib")
        le_path = os.path.join(models_dir, "label_encoder.joblib")
        
        if not os.path.exists(svm_path) or not os.path.exists(le_path):
            raise FileNotFoundError(f"FaceNet models not found in {models_dir}")
        
        self.classifier = joblib.load(svm_path)
        self.label_encoder = joblib.load(le_path)
        
        # Load class centroids for distance-based recognition
        centroids_path = os.path.join(models_dir, 'class_centroids.pkl')
        if os.path.exists(centroids_path):
            import pickle
            try:
                with open(centroids_path, 'rb') as f:
                    self.centroids = pickle.load(f)
                print(f"  ✓ Loaded class centroids for {len(self.centroids)} classes")
            except Exception as e:
                print(f"  ⚠ Warning: Could not load centroids ({e}). Using decision function fallback.")
                self.centroids = None
        else:
            self.centroids = None
        
        # Recognition thresholds
        self.face_recognition_threshold = 0.45
        self.face_recognition_margin = 0.08
        
        print(f"  ✓ FaceNet classes: {list(self.label_encoder.classes_)}")
    
    def _init_behavior_model(self, model_path):
        """Initialize MobileNetV3-Small behavior recognition model"""
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = checkpoint['num_classes']
        
        # Build MobileNetV3-Small
        self.behavior_model = models.mobilenet_v3_small(weights=None)
        in_features = self.behavior_model.classifier[3].in_features
        self.behavior_model.classifier[3] = nn.Linear(in_features, num_classes)
        
        # Load weights
        self.behavior_model.load_state_dict(checkpoint['model_state_dict'])
        self.behavior_model = self.behavior_model.to(self.device)
        self.behavior_model.eval()
        
        # Get class names
        self.behavior_classes = checkpoint.get('class_names', [f'Class_{i}' for i in range(num_classes)])
        
        # Behavior preprocessing
        self.behavior_transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Behavior classification cache
        self.behavior_cache = {}  # {track_id: {'class': name, 'confidence': conf, 'last_frame': num}}
        self.behavior_interval = 10  # Classify every N frames
        
        print(f"  ✓ Behavior classes: {self.behavior_classes}")
    
    def _setup_logging(self):
        """Setup logging directories and CSV"""
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Create directories
        for level in [AUTHORIZED, PARTIAL, UNAUTHORIZED]:
            log_dir = get_log_directory(level)
            ann_dir = get_annotated_directory(level)
            os.makedirs(log_dir, exist_ok=True)
            os.makedirs(ann_dir, exist_ok=True)
        
        # Alert logs
        os.makedirs("logs/main_system/alerts", exist_ok=True)
        os.makedirs("logs/main_system/suspicious_behavior", exist_ok=True)
        
        # CSV for per-frame results
        self.csv_path = f"logs/main_system/tracking_log_{timestamp}.csv"
        os.makedirs(os.path.dirname(self.csv_path), exist_ok=True)
        
        # Write CSV header
        with open(self.csv_path, 'w') as f:
            f.write("timestamp,frame_num,track_id,name,auth_level,behavior_class,behavior_conf,alert_triggered\n")
        
        print(f"  ✓ CSV logging to: {self.csv_path}")
    
    def recognize_face(self, person_crop, original_frame):
        """
        Recognize face within person crop
        
        Returns:
            dict: {'name': str, 'confidence': float} or None if no face found
        """
        if person_crop is None or person_crop.size == 0:
            return None
        
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            
            # Detect faces with MTCNN
            face_boxes, face_probs = self.mtcnn.detect(crop_rgb)
            
            if face_boxes is None or len(face_boxes) == 0:
                return None
            
            # Use highest confidence face
            best_idx = np.argmax(face_probs)
            face_box = face_boxes[best_idx]
            face_prob = face_probs[best_idx]
            
            if face_prob < 0.9:  # Face detection confidence threshold
                return None
            
            # Extract face crop
            x1, y1, x2, y2 = [int(c) for c in face_box]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(crop_rgb.shape[1], x2), min(crop_rgb.shape[0], y2)
            
            if x2 <= x1 or y2 <= y1:
                return None
            
            face_crop = crop_rgb[y1:y2, x1:x2]
            
            # Get embedding
            face_pil = Image.fromarray(face_crop)
            face_tensor = transforms.functional.to_tensor(face_pil).unsqueeze(0).to(self.device)
            
            with torch.no_grad():
                embedding = self.embedder(face_tensor).cpu().numpy()[0]
            
            # Classify
            prediction = self.classifier.predict([embedding])[0]
            name = self.label_encoder.inverse_transform([prediction])[0]
            
            # Calculate confidence using centroid distance if available
            if self.centroids is not None and name in self.centroids:
                centroid = self.centroids[name]
                distance = np.linalg.norm(embedding - centroid)
                confidence = max(0.0, 1.0 - distance)
            else:
                # Use decision function
                decision = self.classifier.decision_function([embedding])
                confidence = float(np.max(decision))
            
            # Apply threshold
            if confidence < self.face_recognition_threshold:
                name = "Unknown"
                confidence = 1.0 - confidence
            
            return {'name': name, 'confidence': confidence}
            
        except Exception as e:
            print(f"[ERROR] Face recognition failed: {e}")
            return None
    
    def classify_behavior(self, person_crop):
        """
        Classify behavior from person crop
        
        Returns:
            dict: {'class_name': str, 'confidence': float}
        """
        try:
            # Convert BGR to RGB
            crop_rgb = cv2.cvtColor(person_crop, cv2.COLOR_BGR2RGB)
            crop_pil = Image.fromarray(crop_rgb)
            
            # Transform
            crop_tensor = self.behavior_transform(crop_pil).unsqueeze(0).to(self.device)
            
            # Inference
            with torch.no_grad():
                outputs = self.behavior_model(crop_tensor)
                probs = torch.nn.functional.softmax(outputs, dim=1)
                confidence, predicted = torch.max(probs, 1)
            
            class_id = predicted.item()
            class_name = self.behavior_classes[class_id]
            conf = confidence.item()
            
            # Apply confidence threshold (use instance setting)
            if conf < self.min_class_conf:
                class_name = "Neutral"
                conf = 1.0 - conf
            
            return {'class_name': class_name, 'confidence': conf}
            
        except Exception as e:
            print(f"[ERROR] Behavior classification failed: {e}")
            return {'class_name': 'Neutral', 'confidence': 0.0}
    
    def should_reclassify_behavior(self, track_id, current_frame):
        """Check if behavior should be re-classified for this track"""
        if track_id not in self.behavior_cache:
            return True
        
        last_frame = self.behavior_cache[track_id].get('last_frame', 0)
        return (current_frame - last_frame) >= self.behavior_interval
    
    def check_authorized_present(self, frame_identities):
        """
        Check if any authorized person is present in current frame
        
        Args:
            frame_identities: List of identity dicts for current frame
        
        Returns:
            bool: True if at least one authorized person present
        """
        for identity in frame_identities:
            if identity.get('auth_level') == AUTHORIZED:
                return True
        return False
    
    def log_frame_data(self, timestamp, frame_num, track_id, name, auth_level, 
                       behavior_class, behavior_conf, alert_triggered):
        """Log per-frame data to CSV"""
        if not self.save_logs or self.csv_path is None:
            return
        
        try:
            with open(self.csv_path, 'a') as f:
                f.write(f"{timestamp},{frame_num},{track_id},{name},{auth_level},"
                       f"{behavior_class},{behavior_conf:.3f},{alert_triggered}\n")
        except Exception as e:
            print(f"[ERROR] Failed to log to CSV: {e}")
    
    def save_alert_frames(self, frame, person_crop, track_id, name, auth_level, 
                          behavior_class, frame_num):
        """Save full frame and cropped person for alerts"""
        if not self.save_logs:
            return
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        
        # Alert directory
        alert_dir = f"logs/main_system/alerts/{auth_level.lower()}"
        os.makedirs(alert_dir, exist_ok=True)
        
        # Save full frame
        full_path = os.path.join(alert_dir, f"alert_full_{timestamp}_t{track_id}_f{frame_num}.jpg")
        cv2.imwrite(full_path, frame)
        
        # Save person crop
        crop_path = os.path.join(alert_dir, f"alert_crop_{timestamp}_t{track_id}_f{frame_num}.jpg")
        cv2.imwrite(crop_path, person_crop)
        
        print(f"  [ALERT SAVED] {name} ({auth_level}) - {behavior_class}")
    
    def can_send_alert(self, track_id):
        """Check if alert cooldown has passed for this track"""
        if track_id not in self.alert_cooldown:
            return True
        
        elapsed = time.time() - self.alert_cooldown[track_id]
        return elapsed >= self.alert_cooldown_seconds
    
    def send_alert(self, track_id, name, auth_level, behavior_class, frame_num):
        """Send alert (placeholder for UI integration)"""
        timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        alert_msg = {
            'timestamp': timestamp,
            'frame': frame_num,
            'track_id': track_id,
            'name': name,
            'auth_level': auth_level,
            'behavior': behavior_class
        }
        
        self.alerts_sent.append(alert_msg)
        self.alert_cooldown[track_id] = time.time()
        
        # Print to console
        print(f"\n🚨 ALERT: {name} ({auth_level}) - {behavior_class} [Frame {frame_num}, Track {track_id}]")
        
        # TODO: Send to UI/notification system
    
    def process_video(self, video_path, display=True, save_output=None,
                     conf_threshold=0.5, iou_threshold=0.7):
        """
        Process video with integrated recognition
        
        Args:
            video_path: Path to video or 'webcam'
            display: Show output window
            save_output: Path to save output video
            conf_threshold: YOLO confidence threshold
            iou_threshold: YOLO IOU threshold
        """
        # Open video
        if video_path == 'webcam':
            cap = cv2.VideoCapture(0)
        else:
            cap = cv2.VideoCapture(video_path)
        
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"\nProcessing: {video_path}")
        print(f"Resolution: {width}x{height} @ {fps} FPS")
        print(f"Total frames: {total_frames}")
        
        # Video writer
        writer = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"Saving output to: {save_output}")
        
        # Processing loop
        frame_num = 0
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_num += 1
                timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S.%f")[:-3]
                
                # YOLO tracking
                results = self.yolo.track(
                    frame,
                    persist=True,
                    tracker='bytetrack.yaml',
                    classes=[0],  # Person class
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                
                # Process detections
                frame_identities = []
                
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy()
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    confidences = results[0].boxes.conf.cpu().numpy()
                    
                    for box, track_id, conf in zip(boxes, track_ids, confidences):
                        x1, y1, x2, y2 = map(int, box)
                        
                        # Crop person
                        person_crop = frame[y1:y2, x1:x2]
                        
                        if person_crop.size == 0:
                            continue
                        
                        # Face recognition
                        face_result = self.recognize_face(person_crop, frame)
                        
                        if face_result is None:
                            name = "Unknown"
                            face_conf = 0.0
                        else:
                            name = face_result['name']
                            face_conf = face_result['confidence']
                        
                        # Get authorization level
                        auth_level = get_authorization_level(name)
                        
                        # Behavior recognition (only if required)
                        behavior_class = "N/A"
                        behavior_conf = 0.0
                        
                        if should_monitor_behavior(auth_level):
                            if self.should_reclassify_behavior(track_id, frame_num):
                                behavior_result = self.classify_behavior(person_crop)
                                behavior_class = behavior_result['class_name']
                                behavior_conf = behavior_result['confidence']
                                
                                # Cache behavior
                                self.behavior_cache[track_id] = {
                                    'class': behavior_class,
                                    'confidence': behavior_conf,
                                    'last_frame': frame_num
                                }
                            else:
                                # Use cached behavior
                                cached = self.behavior_cache.get(track_id, {})
                                behavior_class = cached.get('class', 'Neutral')
                                behavior_conf = cached.get('confidence', 0.0)
                        
                        # Store identity for frame
                        identity_info = {
                            'track_id': track_id,
                            'name': name,
                            'face_conf': face_conf,
                            'auth_level': auth_level,
                            'behavior': behavior_class,
                            'behavior_conf': behavior_conf,
                            'bbox': (x1, y1, x2, y2),
                            'crop': person_crop
                        }
                        frame_identities.append(identity_info)
                
                # Check if authorized person present
                authorized_present = self.check_authorized_present(frame_identities)
                
                # Process alerts and logging
                for identity in frame_identities:
                    track_id = identity['track_id']
                    name = identity['name']
                    auth_level = identity['auth_level']
                    behavior_class = identity['behavior']
                    behavior_conf = identity['behavior_conf']
                    
                    alert_triggered = False
                    
                    # UNAUTHORIZED: Always log and alert (they shouldn't be there)
                    if auth_level == UNAUTHORIZED:
                        if self.can_send_alert(track_id):
                            # Send alert
                            self.send_alert(track_id, name, auth_level, behavior_class, frame_num)
                            alert_triggered = True
                        
                        # Always save frames for unauthorized persons
                        if self.save_logs:
                            self.save_alert_frames(
                                frame, identity['crop'], track_id, name, 
                                auth_level, behavior_class, frame_num
                            )
                    
                    # PARTIAL: Conditional alerts, but always log suspicious behavior
                    elif auth_level == PARTIAL and is_suspicious_behavior(behavior_class):
                        if should_trigger_alert(auth_level, behavior_class, authorized_present):
                            # No authorized present - send alert
                            if self.can_send_alert(track_id):
                                self.send_alert(track_id, name, auth_level, behavior_class, frame_num)
                                alert_triggered = True
                        
                        # Always log suspicious behavior (even if alert suppressed by authorized presence)
                        if self.save_logs:
                            if alert_triggered:
                                # Use alert directory
                                self.save_alert_frames(
                                    frame, identity['crop'], track_id, name, 
                                    auth_level, behavior_class, frame_num
                                )
                            else:
                                # Use suspicious behavior directory (authorized present, no alert)
                                suspicious_dir = "logs/main_system/suspicious_behavior"
                                os.makedirs(suspicious_dir, exist_ok=True)
                                ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                
                                # Save BOTH full frame and cropped frame
                                full_path = os.path.join(suspicious_dir, 
                                                        f"suspicious_full_{ts}_t{track_id}_f{frame_num}.jpg")
                                cv2.imwrite(full_path, frame)
                                
                                crop_path = os.path.join(suspicious_dir, 
                                                        f"suspicious_crop_{ts}_t{track_id}_f{frame_num}.jpg")
                                cv2.imwrite(crop_path, identity['crop'])
                    
                    # Log to CSV
                    self.log_frame_data(
                        timestamp, frame_num, track_id, name, auth_level,
                        behavior_class, behavior_conf, alert_triggered
                    )
                    
                    # Annotate frame
                    x1, y1, x2, y2 = identity['bbox']
                    color = get_level_color(auth_level)
                    thickness = get_level_thickness(auth_level)
                    
                    # Draw box
                    cv2.rectangle(frame, (x1, y1), (x2, y2), color, thickness)
                    
                    # Draw label
                    label = format_display_name(name, auth_level, behavior_class if should_monitor_behavior(auth_level) else None)
                    label_with_conf = f"{label} ({identity['face_conf']:.2f})"
                    
                    # Background for text
                    (text_w, text_h), _ = cv2.getTextSize(label_with_conf, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                    cv2.rectangle(frame, (x1, y1 - text_h - 10), (x1 + text_w, y1), color, -1)
                    cv2.putText(frame, label_with_conf, (x1, y1 - 5), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                    
                    # Show behavior confidence if monitored
                    if should_monitor_behavior(auth_level) and behavior_class != "N/A":
                        behavior_text = f"{behavior_class}: {behavior_conf:.2f}"
                        cv2.putText(frame, behavior_text, (x1, y2 + 20),
                                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
                
                # Draw info overlay
                info_text = f"Frame: {frame_num}/{total_frames} | Auth Present: {authorized_present} | Tracks: {len(frame_identities)}"
                cv2.putText(frame, info_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
                
                # Write frame
                if writer:
                    writer.write(frame)
                
                # Display
                if display:
                    cv2.imshow('Integrated Recognition System', frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                # Progress
                if frame_num % 30 == 0:
                    elapsed = time.time() - start_time
                    current_fps = frame_num / elapsed if elapsed > 0 else 0
                    print(f"  Processed {frame_num}/{total_frames} frames @ {current_fps:.1f} FPS")
        
        finally:
            cap.release()
            if writer:
                writer.release()
            if display:
                cv2.destroyAllWindows()
        
        # Final statistics
        elapsed = time.time() - start_time
        avg_fps = frame_num / elapsed if elapsed > 0 else 0
        
        print("\n" + "="*80)
        print("PROCESSING COMPLETE")
        print("="*80)
        print(f"Total frames: {frame_num}")
        print(f"Total time: {elapsed:.2f}s")
        print(f"Average FPS: {avg_fps:.2f}")
        print(f"Alerts sent: {len(self.alerts_sent)}")
        
        if self.save_logs:
            print(f"CSV log: {self.csv_path}")
        
        print("="*80 + "\n")


def main():
    parser = argparse.ArgumentParser(
        description='Integrated Face Recognition + Behavior Recognition System'
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    
    # Models
    parser.add_argument('--yolo-model', type=str,
                       default='models/YOLOv8/yolov8m.pt',
                       help='Path to YOLOv8 model')
    parser.add_argument('--facenet-model', type=str,
                       default='models/FaceNet/inception_resnet_v1.pt',
                       help='Path to FaceNet model')
    parser.add_argument('--behavior-model', type=str,
                       default='models/mobilenetv3-small/mobilenetv3_small_feature_extraction.pth',
                       help='Path to MobileNetV3-Small behavior model')
    
    # Configuration
    parser.add_argument('--conf', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='YOLO IOU threshold (default: 0.7)')
    
    # Output
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output window')
    parser.add_argument('--save', type=str,
                       help='Save output video to specified path')
    parser.add_argument('--save-logs', action='store_true',
                       help='Save logs and alert frames')
    
    # Device
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    parser.add_argument('--min-class-conf', type=float, default=None,
                       help='Minimum behavior classification confidence (0-1) to accept behavior; below this will be treated as Neutral')
    
    args = parser.parse_args()
    
    # Print authorization configuration
    print_authorization_summary()
    
    # Initialize system
    system = IntegratedRecognitionSystem(
        yolo_model_path=args.yolo_model,
        facenet_model_path=args.facenet_model,
        behavior_model_path=args.behavior_model,
        device=args.device,
        save_logs=args.save_logs,
        min_class_conf=args.min_class_conf
    )
    
    # Process video
    video_source = 'webcam' if args.webcam else args.video
    
    system.process_video(
        video_path=video_source,
        display=not args.no_display,
        save_output=args.save,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )


if __name__ == '__main__':
    main()
