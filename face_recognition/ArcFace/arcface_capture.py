"""Face capture utility for ArcFace training dataset collection using MTCNN."""

import os
import sys
import time
import cv2
import numpy as np
import argparse
from datetime import datetime
from typing import Optional, Tuple, List
from pathlib import Path

# Try to import MTCNN (FaceNet style detection)
try:
    from facenet_pytorch import MTCNN
    import torch
    HAS_MTCNN = True
    print("[INFO] Using MTCNN for face detection")
except ImportError:
    HAS_MTCNN = False
    print("[INFO] MTCNN not available, falling back to Haar cascades")

# Add the root project dir to sys.path for utils
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), "..", "..")))

try:
    from utils.common import align_face, compute_blur_score, estimate_brightness, eye_angle_deg
    HAS_UTILS = True
except ImportError:
    HAS_UTILS = False
    print("[INFO] Utils not available, using basic quality metrics")


class ArcFaceCapture:
    """Face capture utility with MTCNN detection and quality filtering (FaceNet style)."""
    
    def __init__(
        self,
        save_dir: str = None,
        img_size: Tuple[int, int] = (112, 112),
        blur_threshold: float = 35.0,
        brightness_range: Tuple[int, int] = (60, 200),
        max_tilt_deg: float = 20.0,
        min_face_size: int = 60,
        augment_hflip: bool = True
    ):
        """
        Initialize ArcFace capture system with FaceNet-style parameters.
        
        Args:
            save_dir: Directory to save captured faces (defaults to project faces folder)
            img_size: Target image size for saved faces
            blur_threshold: Minimum blur score (higher = sharper required)
            brightness_range: Acceptable brightness range (V channel mean)
            max_tilt_deg: Maximum allowed eye tilt in degrees
            min_face_size: Minimum face size for detection
            augment_hflip: Whether to save horizontally flipped versions
        """
        # Setup save directory
        if save_dir is None:
            project_root = Path(__file__).parent.parent.parent
            save_dir = project_root / 'datasets' / 'faces'
        
        self.save_dir = Path(save_dir)
        self.save_dir.mkdir(parents=True, exist_ok=True)
        
        # Parameters (matching FaceNet capture style)
        self.img_size = img_size
        self.blur_threshold = blur_threshold
        self.brightness_range = brightness_range
        self.max_tilt_deg = max_tilt_deg
        self.min_face_size = min_face_size
        self.augment_hflip = augment_hflip
        
        # Setup device
        if HAS_MTCNN:
            self.device = 'cuda' if torch.cuda.is_available() else 'cpu'
            if self.device == 'cuda':
                torch.backends.cudnn.benchmark = True
            print(f"[INFO] Using device: {self.device}")
        else:
            self.device = 'cpu'
        
        # Initialize face detector
        self._init_detector()
        
        print(f"✅ ArcFace capture initialized")
        print(f"   Save directory: {self.save_dir}")
        print(f"   Target image size: {self.img_size}")
        print(f"   Blur threshold: {self.blur_threshold}")
        print(f"   Brightness range: {self.brightness_range}")
        print(f"   Max tilt: {self.max_tilt_deg}°")
    
    def _init_detector(self):
        """Initialize face detector (MTCNN preferred, Haar cascade fallback)."""
        if HAS_MTCNN:
            # Use MTCNN like FaceNet capture
            self.mtcnn = MTCNN(
                image_size=self.img_size[0],
                margin=0,
                keep_all=True,
                device=self.device,
                post_process=False,
                min_face_size=self.min_face_size
            )
            self.detector_type = 'mtcnn'
            print(f"[INFO] MTCNN detector initialized")
        else:
            # Fallback to Haar cascade
            cascade_path = cv2.data.haarcascades + 'haarcascade_frontalface_default.xml'
            self.face_cascade = cv2.CascadeClassifier(cascade_path)
            self.detector_type = 'haar'
            print(f"[INFO] Haar cascade detector initialized")
    
    def detect_faces_mtcnn(self, frame_rgb: np.ndarray) -> List[dict]:
        """
        Detect faces using MTCNN (FaceNet style).
        
        Args:
            frame_rgb: Input frame in RGB format
            
        Returns:
            List of detection dictionaries with box and keypoints
        """
        detections = []
        
        try:
            boxes, probs, landmarks = self.mtcnn.detect(frame_rgb, landmarks=True)
            
            if boxes is not None:
                for i, box in enumerate(boxes):
                    x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                    
                    keypoints = {}
                    if landmarks is not None and i < len(landmarks):
                        leye = landmarks[i][0]
                        reye = landmarks[i][1]
                        keypoints = {
                            "left_eye": (int(leye[0]), int(leye[1])),
                            "right_eye": (int(reye[0]), int(reye[1]))
                        }
                    
                    detections.append({
                        "box": (x1, y1, x2 - x1, y2 - y1),
                        "keypoints": keypoints,
                        "confidence": float(probs[i]) if probs is not None else 1.0
                    })
        
        except Exception as e:
            print(f"[ERROR] MTCNN detection failed: {e}")
        
        return detections
    
    def detect_faces_haar(self, frame_bgr: np.ndarray) -> List[dict]:
        """
        Detect faces using Haar cascade (fallback).
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            List of detection dictionaries
        """
        gray = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2GRAY)
        
        faces = self.face_cascade.detectMultiScale(
            gray,
            scaleFactor=1.1,
            minNeighbors=5,
            minSize=(self.min_face_size, self.min_face_size),
            flags=cv2.CASCADE_SCALE_IMAGE
        )
        
        detections = []
        for (x, y, w, h) in faces:
            detections.append({
                "box": (x, y, w, h),
                "keypoints": {},
                "confidence": 1.0
            })
        
        return detections
    
    def detect_faces(self, frame_bgr: np.ndarray) -> List[dict]:
        """
        Detect faces using available detector.
        
        Args:
            frame_bgr: Input frame in BGR format
            
        Returns:
            List of detection dictionaries
        """
        if self.detector_type == 'mtcnn':
            frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
            return self.detect_faces_mtcnn(frame_rgb)
        else:
            return self.detect_faces_haar(frame_bgr)
    
    def calculate_face_quality(self, face_region: np.ndarray, keypoints: dict = None) -> Tuple[float, dict]:
        """
        Calculate face quality using FaceNet-style metrics.
        
        Args:
            face_region: Cropped face region (BGR)
            keypoints: Eye keypoints if available
            
        Returns:
            Tuple of (overall_quality, metrics_dict)
        """
        if face_region.size == 0:
            return 0.0, {}
        
        metrics = {}
        
        # Calculate blur score
        if HAS_UTILS:
            blur_score = compute_blur_score(face_region)
            brightness = estimate_brightness(face_region)
            
            if keypoints and "left_eye" in keypoints and "right_eye" in keypoints:
                tilt_deg = abs(eye_angle_deg(keypoints))
            else:
                tilt_deg = 0.0
        else:
            # Fallback blur calculation
            gray = cv2.cvtColor(face_region, cv2.COLOR_BGR2GRAY)
            blur_score = cv2.Laplacian(gray, cv2.CV_64F).var()
            
            # Fallback brightness
            brightness = gray.mean()
            tilt_deg = 0.0
        
        metrics = {
            'blur_score': blur_score,
            'brightness': brightness,
            'tilt_deg': tilt_deg
        }
        
        # Quality checks (FaceNet style)
        quality_passed = (
            blur_score >= self.blur_threshold and
            self.brightness_range[0] <= brightness <= self.brightness_range[1] and
            tilt_deg <= self.max_tilt_deg
        )
        
        overall_quality = 1.0 if quality_passed else 0.0
        
        return overall_quality, metrics
    
    def align_and_preprocess_face(self, frame_bgr: np.ndarray, detection: dict) -> Optional[np.ndarray]:
        """
        Align and preprocess face (FaceNet style).
        
        Args:
            frame_bgr: Full frame in BGR
            detection: Detection dictionary with box and keypoints
            
        Returns:
            Aligned face image or None if failed
        """
        x, y, w, h = detection['box']
        keypoints = detection.get('keypoints', {})
        
        # Try alignment if we have eye keypoints
        eyes_found = "left_eye" in keypoints and "right_eye" in keypoints
        aligned = None
        
        if eyes_found and HAS_UTILS:
            try:
                aligned = align_face(frame_bgr, (x, y, w, h), keypoints, output_size=self.img_size)
            except Exception as e:
                print(f"[DEBUG] Alignment failed: {e}")
                aligned = None
        
        # Fallback to simple crop and resize
        if aligned is None:
            face_crop = frame_bgr[max(0, y):y + h, max(0, x):x + w]
            if face_crop.size == 0:
                return None
            aligned = cv2.resize(face_crop, self.img_size)
        
        return aligned
    
    def save_face(self, face_image: np.ndarray, person_name: str, count: int) -> str:
        """
        Save face image (FaceNet style naming).
        
        Args:
            face_image: Processed face image
            person_name: Name of the person
            count: Image count/index
            
        Returns:
            Path to saved image
        """
        # Create person directory
        person_dir = self.save_dir / person_name
        person_dir.mkdir(exist_ok=True)
        
        # FaceNet style filename
        filename = f"{person_name}_{count}.jpg"
        filepath = person_dir / filename
        
        # Save image with high quality
        cv2.imwrite(str(filepath), face_image, [cv2.IMWRITE_JPEG_QUALITY, 95])
        
        return str(filepath)
    
    def capture_from_webcam(
        self,
        person_name: str,
        max_images: int = 100,
        camera_id: int = 0
    ) -> List[str]:
        """
        Capture faces from webcam (FaceNet style interface).
        
        Args:
            person_name: Name of the person to capture
            max_images: Maximum number of images to capture
            camera_id: Camera device ID
            
        Returns:
            List of saved image paths
        """
        print(f"\n📹 Starting ArcFace capture for: {person_name}")
        print(f"   Max images: {max_images}")
        print(f"   Camera ID: {camera_id}")
        print(f"   Save directory: {self.save_dir / person_name}")
        print(f"[INFO] Press 'q' to quit early.")
        
        # Clean person name (replace spaces with underscores)
        person_name = person_name.strip().replace(" ", "_")
        
        # Open camera
        cap = cv2.VideoCapture(camera_id)
        if HAS_MTCNN:
            cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)
        
        if not cap.isOpened():
            raise ValueError(f"[ERROR] Webcam not found: {camera_id}")
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 640)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 480)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        print(f"[INFO] Get ready! Capturing faces for: {person_name}")
        
        # Countdown with warm-up (FaceNet style)
        for i in range(5, 0, -1):
            ret, frame = cap.read()
            if not ret:
                print("[ERROR] Failed to grab frame during countdown.")
                break
            
            # Warm-up detection (forces CUDA kernels to load)
            if self.detector_type == 'mtcnn':
                frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
                _ = self.mtcnn.detect(frame_rgb)
            
            cv2.putText(frame, f"Starting in {i}...", (50, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 2, (0, 255, 255), 3)
            cv2.imshow("ArcFace Capture", frame)
            cv2.waitKey(1000)
        
        count = 0
        saved_paths = []
        
        print(f"[INFO] Starting capture... Look at the camera!")
        
        try:
            while count < max_images:
                ret, frame_bgr = cap.read()
                if not ret:
                    print("[ERROR] Failed to grab frame.")
                    break
                
                # Detect faces
                detections = self.detect_faces(frame_bgr)
                
                # Process each detection
                for detection in detections:
                    if count >= max_images:
                        break
                    
                    # Align and preprocess face
                    aligned = self.align_and_preprocess_face(frame_bgr, detection)
                    if aligned is None:
                        continue
                    
                    # Check quality
                    quality, metrics = self.calculate_face_quality(aligned, detection.get('keypoints'))
                    
                    if quality <= 0:
                        continue  # Skip low quality faces
                    
                    # Save aligned face
                    saved_path = self.save_face(aligned, person_name, count)
                    saved_paths.append(saved_path)
                    
                    print(f"[INFO] Saved {person_name}_{count}.jpg "
                          f"(blur={metrics.get('blur_score', 0):.1f}, "
                          f"bright={metrics.get('brightness', 0):.1f}, "
                          f"tilt={metrics.get('tilt_deg', 0):.1f})")
                    count += 1
                    
                    # Optional horizontal flip augmentation (FaceNet style)
                    if count < max_images and self.augment_hflip:
                        flipped = cv2.flip(aligned, 1)
                        saved_path = self.save_face(flipped, person_name, count)
                        saved_paths.append(saved_path)
                        print(f"[INFO] Saved augmented {person_name}_{count}.jpg")
                        count += 1
                
                # Draw detections on frame
                for detection in detections:
                    x, y, w, h = detection['box']
                    cv2.rectangle(frame_bgr, (x, y), (x + w, y + h), (0, 255, 0), 2)
                
                # Show progress
                cv2.putText(frame_bgr, f"Captured: {count}/{max_images}", (10, 30),
                           cv2.FONT_HERSHEY_SIMPLEX, 0.8, (255, 255, 255), 2)
                cv2.imshow("ArcFace Capture", frame_bgr)
                
                # Check for quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("[INFO] Early exit requested.")
                    break
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
        
        print(f"[DONE] Collected {count} face images in: {self.save_dir / person_name}")
        return saved_paths


def main():
    """Command-line interface matching FaceNet capture style."""
    parser = argparse.ArgumentParser(description='ArcFace Capture (FaceNet Style)')
    parser.add_argument('--name', type=str, help='Name of the person to capture (interactive if not provided)')
    parser.add_argument('--max_images', type=int, default=100, help='Maximum number of images to capture')
    parser.add_argument('--camera', type=int, default=0, help='Camera device ID')
    parser.add_argument('--save_dir', type=str, help='Directory to save faces')
    parser.add_argument('--blur_threshold', type=float, default=35.0, help='Minimum blur threshold')
    parser.add_argument('--brightness_min', type=int, default=60, help='Minimum brightness')
    parser.add_argument('--brightness_max', type=int, default=200, help='Maximum brightness')
    parser.add_argument('--max_tilt', type=float, default=20.0, help='Maximum tilt in degrees')
    parser.add_argument('--no_flip', action='store_true', help='Disable horizontal flip augmentation')
    
    args = parser.parse_args()
    
    # Get person name (interactive like FaceNet)
    if args.name:
        person_name = args.name
    else:
        person_name = input("Enter name: ").strip()
    
    if not person_name:
        print("Error: Person name is required")
        return
    
    # Initialize capture
    capture = ArcFaceCapture(
        save_dir=args.save_dir,
        blur_threshold=args.blur_threshold,
        brightness_range=(args.brightness_min, args.brightness_max),
        max_tilt_deg=args.max_tilt,
        augment_hflip=not args.no_flip
    )
    
    try:
        saved_paths = capture.capture_from_webcam(
            person_name=person_name,
            max_images=args.max_images,
            camera_id=args.camera
        )
        print(f"\n✅ Captured {len(saved_paths)} images for {person_name}")
    
    except KeyboardInterrupt:
        print(f"\n⏹️  Capture interrupted by user")
    except Exception as e:
        print(f"\n❌ Capture failed: {e}")


if __name__ == "__main__":
    main()