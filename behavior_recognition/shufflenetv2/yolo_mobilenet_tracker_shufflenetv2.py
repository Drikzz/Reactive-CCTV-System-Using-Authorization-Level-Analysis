"""
YOLOv8 + ShuffleNetV2 Integrated Tracking and Classification System

This script combines:
- YOLOv8 object detection and tracking (ByteTrack)
- ShuffleNetV2 behavior classification
- Adaptive classification intervals based on FPS
- Per-track classification caching

Features:
- Tracks multiple people in the frame
- Crops each tracked person for classification
- Caches classifications per track_id
- Re-classifies only every N frames (adaptive based on FPS)
- Annotates with track_id, class, and confidence
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import time
from collections import deque, defaultdict
from ultralytics import YOLO

# Add parent directory to path for imports (now two levels up since we're in shufflenetv2 subfolder)
BASE_DIR = Path(__file__).parent.parent.parent
sys.path.append(str(BASE_DIR))


class YOLOShuffleNetV2Tracker:
    """
    Integrated YOLOv8 tracking + ShuffleNetV2 classification system.
    """
    
    def __init__(self, yolo_model_path, mobilenet_model_path, class_names=None, 
                 tracker="bytetrack.yaml", device=None, reclass_interval=10, smoothing_window=7):
        """
        Initialize the integrated tracker.
        
        Args:
            yolo_model_path (str): Path to YOLOv8 model
            mobilenet_model_path (str): Path to trained ShuffleNetV2 checkpoint
            class_names (list): List of behavior class names
            tracker (str): Tracker configuration file (default: bytetrack.yaml)
            device (str): Device to use ('cuda' or 'cpu')
            reclass_interval (int): Initial re-classification interval in frames
            smoothing_window (int): Temporal smoothing window size (recent predictions to average)
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.tracker_name = tracker
        self.initial_reclass_interval = reclass_interval
        self.smoothing_window = max(1, int(smoothing_window))
        
        # Load YOLOv8 model
        print(f"Loading YOLOv8 model from: {yolo_model_path}")
        self.yolo = YOLO(yolo_model_path)
        
        # Load ShuffleNetV2 model
        print(f"Loading ShuffleNetV2 model from: {mobilenet_model_path}")
        self.mobilenet, self.num_classes = self._load_shufflenet_v2(mobilenet_model_path)
        
        # Set class names
        if class_names is None:
            # Try to load from checkpoint
            checkpoint = torch.load(mobilenet_model_path, map_location=self.device)
            self.class_names = checkpoint.get('class_names', [f"Class_{i}" for i in range(self.num_classes)])
        else:
            self.class_names = class_names
        
        # Define transform for ShuffleNetV2
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        # Classification cache: {track_id: {'class': name, 'confidence': conf, 'last_frame': num}}
        self.classification_cache = {}
        # Temporal smoothing: recent probability vectors per track
        self.prob_history = defaultdict(lambda: deque(maxlen=self.smoothing_window))
        
        # 🧠 Adaptive classification setup
        self.fps_window = deque(maxlen=10)
        self.reclass_interval = reclass_interval
        # Dynamic minimum classification confidence (0-1). Predictions below this will be labeled 'Neutral'
        self.min_class_conf = 0.6
        self.last_reclass_update = time.time()
        self.reclass_update_interval = 2.0  # Update every 2 seconds
        
        print(f"\n✓ YOLOv8 Tracker: {tracker}")
        print(f"✓ ShuffleNetV2 Classes: {self.class_names}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Initial Classification Interval: every {reclass_interval} frames")
        print(f"🧠 Adaptive Classification: ENABLED\n")
    
    def _load_shufflenet_v2(self, model_path):
        """Load ShuffleNetV2 model from checkpoint."""
        checkpoint = torch.load(model_path, map_location=self.device)
        num_classes = checkpoint['num_classes']
        
        # Build model architecture - ShuffleNetV2 x1.0
        model = models.shufflenet_v2_x1_0(weights=None)
        in_features = model.fc.in_features
        model.fc = nn.Linear(in_features, num_classes)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model, num_classes
    
    def _preprocess_crop(self, crop):
        """Preprocess a person crop for ShuffleNetV2."""
        # Convert BGR to RGB
        crop_rgb = cv2.cvtColor(crop, cv2.COLOR_BGR2RGB)
        crop_pil = Image.fromarray(crop_rgb)
        
        # Apply transforms
        crop_tensor = self.transform(crop_pil)
        crop_tensor = crop_tensor.unsqueeze(0)
        
        return crop_tensor
    
    def _classify_crop(self, crop):
        """
        Classify a person crop using ShuffleNetV2.
        
        Args:
            crop: OpenCV image crop (BGR)
            
        Returns:
            dict: {'class_name': str, 'class_id': int, 'confidence': float, 'probs': np.ndarray}
        """
        # Preprocess
        crop_tensor = self._preprocess_crop(crop)
        crop_tensor = crop_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.mobilenet(crop_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
            probs_np = probabilities.squeeze(0).detach().cpu().numpy()

        # Get raw results
        class_id = predicted.item()
        confidence_score = confidence.item()

        # Apply dynamic threshold: if confidence below min_class_conf, mark as Unknown
        if confidence_score < self.min_class_conf:
            return {
                'class_name': 'Neutral',
                'class_id': -1,
                'confidence': confidence_score,
                'probs': probs_np
            }

        class_name = self.class_names[class_id]
        return {
            'class_name': class_name,
            'class_id': class_id,
            'confidence': confidence_score,
            'probs': probs_np
        }
    
    def _should_reclassify(self, track_id, current_frame):
        """
        Check if we should re-classify this track.
        
        Args:
            track_id: Track identifier
            current_frame: Current frame number
            
        Returns:
            bool: True if should classify
        """
        if track_id not in self.classification_cache:
            return True  # First time seeing this track
        
        last_classified = self.classification_cache[track_id]['last_frame']
        frames_since_last = current_frame - last_classified
        
        return frames_since_last >= self.reclass_interval
    
    def _update_adaptive_interval(self, current_fps):
        """
        Update classification interval based on current FPS.
        
        Args:
            current_fps: Current processing FPS
        """
        current_time = time.time()
        if current_time - self.last_reclass_update > self.reclass_update_interval:
            old_interval = self.reclass_interval
            
            # Adaptive interval based on FPS ranges
            if current_fps > 25:
                # High FPS: classify more frequently
                self.reclass_interval = max(5, self.initial_reclass_interval // 2)
            elif current_fps >= 15:
                # Medium FPS: balanced classification
                self.reclass_interval = self.initial_reclass_interval
            elif current_fps >= 10:
                # Low FPS: classify less frequently
                self.reclass_interval = min(30, self.initial_reclass_interval * 2)
            else:
                # Very low FPS: classify even less to maintain smoothness
                self.reclass_interval = min(40, self.initial_reclass_interval * 3)
            
            # Log interval changes
            if old_interval != self.reclass_interval:
                print(f"🔄 FPS: {current_fps:.1f} → Adjusted interval: {old_interval} → {self.reclass_interval} frames")
            
            self.last_reclass_update = current_time
    
    def _annotate_frame(self, frame, tracks_data, current_fps=None):
        """
        Annotate frame with tracking and classification results.
        
        Args:
            frame: OpenCV frame
            tracks_data: List of dicts with track info
            current_fps: Current processing FPS
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Draw each tracked person
        for track_info in tracks_data:
            track_id = track_info['track_id']
            bbox = track_info['bbox']
            class_name = track_info['class_name']
            confidence = track_info['confidence']
            
            x1, y1, x2, y2 = bbox
            
            # Choose color based on track_id
            colors = [
                (0, 255, 0),    # Green
                (255, 0, 0),    # Blue
                (0, 255, 255),  # Yellow
                (255, 0, 255),  # Magenta
                (255, 128, 0),  # Orange
                (128, 0, 255),  # Purple
            ]
            color = colors[track_id % len(colors)]
            
            # Draw bounding box
            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
            
            # Create label with background
            label = f"ID:{track_id} | {class_name} ({confidence*100:.1f}%)"
            
            # Get label size
            font = cv2.FONT_HERSHEY_SIMPLEX
            font_scale = 0.6
            thickness = 2
            (label_width, label_height), baseline = cv2.getTextSize(label, font, font_scale, thickness)
            
            # Draw label background
            label_y = max(y1 - 10, label_height + 10)
            cv2.rectangle(annotated_frame, 
                         (x1, label_y - label_height - 5),
                         (x1 + label_width + 5, label_y + baseline),
                         color, -1)
            
            # Draw label text
            cv2.putText(annotated_frame, label, (x1 + 2, label_y - 5),
                       font, font_scale, (255, 255, 255), thickness)
        
        # Add global info overlay
        overlay = annotated_frame.copy()
        info_height = 100 if current_fps is not None else 70
        cv2.rectangle(overlay, (10, height - info_height - 10), 
                     (width - 10, height - 10), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        # Draw global info
        font = cv2.FONT_HERSHEY_SIMPLEX
        y_offset = height - info_height + 5
        
        cv2.putText(annotated_frame, f"Tracked People: {len(tracks_data)}",
                   (20, y_offset + 20), font, 0.6, (255, 255, 255), 2)
        
        cv2.putText(annotated_frame, f"Classification Interval: every {self.reclass_interval} frames",
                   (20, y_offset + 45), font, 0.6, (100, 200, 255), 2)
        
        if current_fps is not None:
            cv2.putText(annotated_frame, f"Processing FPS: {current_fps:.1f}",
                       (20, y_offset + 70), font, 0.6, (100, 200, 255), 2)
        # Show current minimum classification confidence and smoothing window
        cv2.putText(annotated_frame, f"Min Class Conf: {self.min_class_conf:.2f}",
                   (width - 240, y_offset + 20), font, 0.6, (200, 200, 100), 2)
        cv2.putText(annotated_frame, f"Smoothing Win: {self.smoothing_window}",
                   (width - 240, y_offset + 45), font, 0.6, (200, 200, 100), 2)
        
        return annotated_frame
    
    def process_video(self, video_path, display=True, save_output=None, 
                     conf_threshold=0.5, iou_threshold=0.7):
        """
        Process video with YOLOv8 tracking + ShuffleNetV2 classification.
        
        Args:
            video_path (str): Path to video file or 'webcam' for webcam
            display (bool): Whether to display output
            save_output (str): Path to save output video
            conf_threshold (float): YOLO confidence threshold
            iou_threshold (float): YOLO IOU threshold
            
        Returns:
            dict: Processing statistics
        """
        # Open video
        if video_path == 'webcam':
            cap = cv2.VideoCapture(0)
            is_webcam = True
            print("✓ Webcam opened")
        else:
            cap = cv2.VideoCapture(str(video_path))
            is_webcam = False
            print(f"✓ Video opened: {video_path}")
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else None
        
        print(f"Video Info: {width}x{height} @ {fps} FPS")
        if total_frames:
            print(f"Total Frames: {total_frames}")
        print()
        
        # Setup video writer if saving output
        out = None
        if save_output:
            save_path = Path(save_output)
            if not save_path.parent.name or save_path.parent == Path('.'):
                save_path = Path('outputs') / save_path.name
            
            if not save_path.is_absolute():
                save_path = BASE_DIR / save_path
            
            save_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(str(save_path), fourcc, fps, (width, height))
            
            if not out.isOpened():
                print(f"⚠️ Warning: Could not create video writer")
                out = None
            else:
                print(f"✓ Saving output to: {save_path}\n")
        
        # Statistics
        frame_count = 0
        total_detections = 0
        total_classifications = 0
        classification_times = []
        
        print("Processing video...")
        print("Press 'q' to quit")
        print("Press '[' to decrease confidence threshold")
        print("Press ']' to increase confidence threshold\n")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                frame_count += 1
                frame_start_time = time.time()
                
                # 🎯 YOLOv8 Tracking (detect and track people)
                results = self.yolo.track(
                    frame,
                    persist=True,
                    tracker=self.tracker_name,
                    classes=[0],  # Person class only
                    conf=conf_threshold,
                    iou=iou_threshold,
                    verbose=False
                )
                
                tracks_data = []
                
                # Process each tracked person
                if results[0].boxes is not None and results[0].boxes.id is not None:
                    boxes = results[0].boxes.xyxy.cpu().numpy().astype(int)
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                    
                    total_detections += len(track_ids)
                    
                    for box, track_id in zip(boxes, track_ids):
                        x1, y1, x2, y2 = box
                        
                        # Ensure valid crop coordinates
                        x1 = max(0, x1)
                        y1 = max(0, y1)
                        x2 = min(width, x2)
                        y2 = min(height, y2)
                        
                        # Skip if invalid box
                        if x2 <= x1 or y2 <= y1:
                            continue
                        
                        # 🔍 Check if we should classify this track
                        if self._should_reclassify(track_id, frame_count):
                            # Crop person
                            person_crop = frame[y1:y2, x1:x2]
                            
                            # 🤖 Classify with ShuffleNetV2
                            classify_start = time.time()
                            result_current = self._classify_crop(person_crop)
                            classify_time = time.time() - classify_start
                            classification_times.append(classify_time)
                            
                            # ➿ Temporal smoothing: update history and compute averaged probabilities
                            self.prob_history[track_id].append(result_current['probs'])
                            hist = self.prob_history[track_id]
                            if len(hist) == 1:
                                smoothed_probs = hist[0]
                            else:
                                smoothed_probs = np.mean(np.stack(hist, axis=0), axis=0)
                            
                            smoothed_class_id = int(np.argmax(smoothed_probs))
                            smoothed_confidence = float(smoothed_probs[smoothed_class_id])
                            
                            if smoothed_confidence < self.min_class_conf:
                                result = {
                                    'class_name': 'Neutral',
                                    'class_id': -1,
                                    'confidence': smoothed_confidence
                                }
                            else:
                                result = {
                                    'class_name': self.class_names[smoothed_class_id],
                                    'class_id': smoothed_class_id,
                                    'confidence': smoothed_confidence
                                }
                            
                            # Update cache
                            self.classification_cache[track_id] = {
                                'class_name': result['class_name'],
                                'confidence': result['confidence'],
                                'last_frame': frame_count
                            }
                            
                            total_classifications += 1
                        else:
                            # Use cached classification
                            result = {
                                'class_name': self.classification_cache[track_id]['class_name'],
                                'confidence': self.classification_cache[track_id]['confidence']
                            }
                        
                        # Store track data
                        tracks_data.append({
                            'track_id': track_id,
                            'bbox': (x1, y1, x2, y2),
                            'class_name': result['class_name'],
                            'confidence': result['confidence']
                        })
                
                # 🧠 Update adaptive interval based on FPS
                frame_end_time = time.time()
                frame_processing_time = frame_end_time - frame_start_time
                self.fps_window.append(frame_processing_time)
                
                if len(self.fps_window) > 0:
                    avg_frame_time = sum(self.fps_window) / len(self.fps_window)
                    current_fps = 1.0 / avg_frame_time if avg_frame_time > 0 else 0
                    self._update_adaptive_interval(current_fps)
                else:
                    current_fps = None
                
                # 🎨 Annotate frame
                annotated_frame = self._annotate_frame(frame, tracks_data, current_fps)
                
                # Display progress
                if not is_webcam and total_frames:
                    progress = (frame_count / total_frames) * 100
                    if frame_count % 30 == 0:  # Print every 30 frames
                        fps_display = current_fps if current_fps else 0.0
                        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                              f"People: {len(tracks_data)} - "
                              f"FPS: {fps_display:.1f} - "
                              f"Interval: {self.reclass_interval}")
                
                # Display
                if display:
                    cv2.imshow('YOLOv8 + ShuffleNetV2 Tracking', annotated_frame)
                    key = cv2.waitKey(1) & 0xFF
                    if key == ord('q'):
                        print("\nStopped by user")
                        break
                    # Decrease threshold with '[' and increase with ']' keys
                    if key == ord(']'):
                        old = self.min_class_conf
                        self.min_class_conf = min(0.99, round(self.min_class_conf + 0.05, 2))
                        print(f"🔽 Min class conf: {old:.2f} -> {self.min_class_conf:.2f}")
                    if key == ord('['):
                        old = self.min_class_conf
                        self.min_class_conf = max(0.0, round(self.min_class_conf - 0.05, 2))
                        print(f"🔼 Min class conf: {old:.2f} -> {self.min_class_conf:.2f}")
                
                # Save frame
                if out:
                    out.write(annotated_frame)
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*70)
        print("PROCESSING SUMMARY")
        print("="*70)
        print(f"Total Frames Processed: {frame_count}")
        print(f"Total Person Detections: {total_detections}")
        print(f"Total Classifications: {total_classifications}")
        print(f"Classification Rate: {(total_classifications/frame_count)*100:.1f}% of frames")
        
        if classification_times:
            avg_class_time = np.mean(classification_times)
            print(f"Average Classification Time: {avg_class_time*1000:.2f}ms")
        
        if len(self.fps_window) > 0:
            final_fps = 1.0 / (sum(self.fps_window) / len(self.fps_window))
            print(f"\n🧠 Adaptive Classification Summary:")
            print(f"   - Final Processing FPS: {final_fps:.1f}")
            print(f"   - Final Classification Interval: every {self.reclass_interval} frames")
            print(f"   - Unique Tracks Seen: {len(self.classification_cache)}")
        
        print("="*70 + "\n")
        
        return {
            'frames': frame_count,
            'detections': total_detections,
            'classifications': total_classifications,
            'unique_tracks': len(self.classification_cache)
        }


def main():
    """Main function with command-line interface."""
    parser = argparse.ArgumentParser(
        description='YOLOv8 Tracking + ShuffleNetV2 Classification System'
    )
    
    # Input
    input_group = parser.add_mutually_exclusive_group(required=True)
    input_group.add_argument('--video', type=str, help='Path to input video')
    input_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    
    # Models
    parser.add_argument('--yolo-model', type=str,
                       default='models/YOLOv8/yolov8m.pt',
                       help='Path to YOLOv8 model (default: models/YOLOv8/yolov8m.pt)')
    parser.add_argument('--mobilenet-model', type=str,
                       default='models/shufflenetv2/shufflenet_v2_transfer.pth',
                       help='Path to ShuffleNetV2 model (default: models/shufflenetv2/shufflenet_v2_transfer.pth)')
    
    # Configuration
    parser.add_argument('--tracker', type=str, default='bytetrack.yaml',
                       help='Tracker config file (default: bytetrack.yaml)')
    parser.add_argument('--reclass-interval', type=int, default=10,
                       help='Initial re-classification interval in frames (default: 10)')
    parser.add_argument('--conf', type=float, default=0.5,
                       help='YOLO confidence threshold (default: 0.5)')
    parser.add_argument('--iou', type=float, default=0.7,
                       help='YOLO IOU threshold (default: 0.7)')
    parser.add_argument('--min-class-conf', type=float, default=0.6,
                       help='Minimum classification confidence (0-1) to accept behavior, default 0.6')
    parser.add_argument('--smooth-window', type=int, default=7,
                       help='Temporal smoothing window size in frames (default: 7)')
    
    # Output
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output window')
    parser.add_argument('--save', type=str,
                       help='Save output video to specified path')
    
    # Device
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    # Class names
    parser.add_argument('--classes', type=str, nargs='+',
                       help='Behavior class names (optional, auto-detect from model)')
    
    args = parser.parse_args()
    
    # Resolve model paths
    yolo_path = Path(args.yolo_model)
    if not yolo_path.is_absolute():
        yolo_path = BASE_DIR / yolo_path
    
    mobilenet_path = Path(args.mobilenet_model)
    if not mobilenet_path.is_absolute():
        mobilenet_path = BASE_DIR / mobilenet_path
    
    # Check if models exist
    if not yolo_path.exists():
        print(f"Error: YOLOv8 model not found: {yolo_path}")
        return
    
    if not mobilenet_path.exists():
        print(f"Error: ShuffleNetV2 model not found: {mobilenet_path}")
        return
    
    # Initialize tracker
    print("\n" + "="*70)
    print("YOLOv8 + SHUFFLENETV2 INTEGRATED TRACKER")
    print("="*70 + "\n")
    
    tracker = YOLOShuffleNetV2Tracker(
        yolo_model_path=str(yolo_path),
        mobilenet_model_path=str(mobilenet_path),
        class_names=args.classes,
        tracker=args.tracker,
        device=args.device,
        reclass_interval=args.reclass_interval,
        smoothing_window=args.smooth_window
    )
    # Apply CLI-provided minimum class confidence
    try:
        tracker.min_class_conf = float(args.min_class_conf)
    except Exception:
        print(f"Invalid --min-class-conf value: {args.min_class_conf}. Using default {tracker.min_class_conf}")
    
    # Process video
    video_source = 'webcam' if args.webcam else args.video
    
    tracker.process_video(
        video_path=video_source,
        display=not args.no_display,
        save_output=args.save,
        conf_threshold=args.conf,
        iou_threshold=args.iou
    )
    
    print("Processing complete!")


if __name__ == "__main__":
    main()
