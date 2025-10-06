"""
Google Colab Inference for Behavior Recognition
Optimized for Jupyter/Colab notebooks with inline video display.

Usage in Colab:
    # Upload video
    from google.colab import files
    uploaded = files.upload()
    
    # Run inference
    !python inference_colab.py --video uploaded_video.mp4
    
    # Download result
    from google.colab import files
    files.download('uploaded_video_annotated.mp4')
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO
import argparse
import time
import sys
from tqdm.notebook import tqdm as notebook_tqdm
from tqdm import tqdm as console_tqdm

# Try to import config, handle if not in path
try:
    import config
except ImportError:
    sys.path.insert(0, str(Path(__file__).parent))
    import config

from train_lstm_classifier import LSTMClassifier


class PersonTracker:
    """Tracks individual persons and their pose sequences."""
    
    def __init__(self, sequence_length=60, smoothing_window=5):
        self.sequence_length = sequence_length
        self.smoothing_window = smoothing_window
        
        # Per-person tracking
        self.person_sequences = defaultdict(lambda: deque(maxlen=sequence_length))
        self.person_predictions = defaultdict(lambda: deque(maxlen=smoothing_window))
        self.person_last_seen = {}
        
        # Timeout for removing inactive tracks
        self.track_timeout = 30
    
    def update(self, person_id, keypoints):
        """Update sequence for a tracked person."""
        self.person_sequences[person_id].append(keypoints)
        self.person_last_seen[person_id] = 0
    
    def add_prediction(self, person_id, predicted_class):
        """Add prediction to smoothing buffer."""
        self.person_predictions[person_id].append(predicted_class)
    
    def get_sequence(self, person_id):
        """Get current sequence for person."""
        return list(self.person_sequences[person_id])
    
    def get_smoothed_prediction(self, person_id):
        """Get majority vote prediction."""
        if person_id not in self.person_predictions or len(self.person_predictions[person_id]) == 0:
            return None
        
        predictions = list(self.person_predictions[person_id])
        return max(set(predictions), key=predictions.count)
    
    def is_ready(self, person_id):
        """Check if person has enough frames."""
        return len(self.person_sequences[person_id]) >= self.sequence_length
    
    def cleanup(self):
        """Remove inactive tracks."""
        to_remove = []
        
        for person_id in list(self.person_last_seen.keys()):
            self.person_last_seen[person_id] += 1
            
            if self.person_last_seen[person_id] > self.track_timeout:
                to_remove.append(person_id)
        
        for person_id in to_remove:
            if person_id in self.person_sequences:
                del self.person_sequences[person_id]
            if person_id in self.person_predictions:
                del self.person_predictions[person_id]
            if person_id in self.person_last_seen:
                del self.person_last_seen[person_id]


class ColabBehaviorInference:
    """Colab-optimized behavior classification."""
    
    def __init__(self, model_path, pose_model_path, device='cuda'):
        """Initialize inference system."""
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("=" * 60)
        print("GOOGLE COLAB BEHAVIOR INFERENCE")
        print("=" * 60)
        
        # Load LSTM classifier
        print(f"\n🧠 Loading LSTM classifier...")
        checkpoint = torch.load(model_path, map_location=self.device)
        
        self.class_names = checkpoint['class_names']
        model_config = checkpoint['config']
        
        self.lstm_model = LSTMClassifier(
            input_dim=model_config['input_dim'],
            hidden_dim=model_config['hidden_dim'],
            num_layers=model_config['num_layers'],
            num_classes=model_config['num_classes'],
            dropout=model_config['dropout']
        ).to(self.device)
        
        self.lstm_model.load_state_dict(checkpoint['model_state_dict'])
        self.lstm_model.eval()
        
        print(f"✓ LSTM loaded | Classes: {self.class_names}")
        print(f"✓ Device: {self.device}")
        
        # Load YOLOv8-pose
        print(f"\n📦 Loading YOLOv8-pose...")
        self.pose_model = YOLO(str(pose_model_path))
        self.pose_model.to(self.device)
        print("✓ YOLOv8-pose loaded")
        
        # Initialize tracker
        self.sequence_length = config.FIXED_SEQUENCE_LENGTH or 60
        self.tracker = PersonTracker(sequence_length=self.sequence_length, smoothing_window=5)
        
        # Colors
        self.colors = self._generate_colors(20)
        
        # Stats
        self.frame_count = 0
        self.total_detections = 0
        self.behavior_counts = defaultdict(int)
    
    def _generate_colors(self, n):
        """Generate distinct colors."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def normalize_keypoints(self, keypoints, frame_width, frame_height):
        """Normalize keypoints."""
        kp = keypoints.copy()
        kp[:, 0] = kp[:, 0] / frame_width
        kp[:, 1] = kp[:, 1] / frame_height
        return kp
    
    def classify_sequence(self, sequence):
        """Classify pose sequence."""
        if len(sequence) < self.sequence_length:
            return None, 0.0, None
        
        seq_array = np.array(sequence[-self.sequence_length:])
        seq_array = seq_array.reshape(seq_array.shape[0], -1)
        
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0)
        seq_tensor = seq_tensor.to(self.device)
        
        with torch.no_grad():
            outputs = self.lstm_model(seq_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        return predicted.item(), confidence.item(), probabilities.cpu().numpy()[0]
    
    def draw_skeleton(self, frame, keypoints):
        """Draw pose skeleton."""
        kp = keypoints.astype(int)
        
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],
            [5, 11], [6, 12], [5, 6],
            [5, 7], [6, 8], [7, 9], [8, 10],
            [0, 1], [0, 2], [1, 3], [2, 4]
        ]
        
        for connection in skeleton:
            pt1 = tuple(kp[connection[0]])
            pt2 = tuple(kp[connection[1]])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        for point in kp:
            cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
    
    def draw_info_panel(self, frame, tracked_count):
        """Draw info panel with stats."""
        h, w = frame.shape[:2]
        
        # Background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (350, 120), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.7, frame, 0.3, 0)
        
        # Text
        cv2.putText(frame, f"Frame: {self.frame_count}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        cv2.putText(frame, f"Tracked: {tracked_count}", (20, 60),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        cv2.putText(frame, f"Window: {self.sequence_length} frames", (20, 85),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        # Device info
        device_text = f"Device: {str(self.device).upper()}"
        cv2.putText(frame, device_text, (20, 110),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (100, 200, 100), 1)
        
        return frame
    
    def draw_person_box(self, frame, bbox, person_id, behavior, confidence, color):
        """Draw bounding box and label."""
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 3)
        
        # Label
        if behavior is not None:
            label = f"ID:{person_id} | {behavior} ({confidence*100:.0f}%)"
            label_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        else:
            label = f"ID:{person_id} | Loading..."
            label_color = (150, 150, 150)
        
        # Label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.7, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 15), (x1 + label_w + 10, y1), color, -1)
        
        # Label text
        cv2.putText(frame, label, (x1 + 5, y1 - 8),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    
    def process_frame(self, frame):
        """Process single frame."""
        h, w = frame.shape[:2]
        
        # Track persons
        results = self.pose_model.track(
            frame,
            conf=config.YOLO_CONF_THRESHOLD,
            persist=True,
            verbose=False
        )
        
        tracked_count = 0
        
        # Process detections
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()
                
                for idx, (bbox, person_id) in enumerate(zip(boxes, track_ids)):
                    if idx >= len(keypoints):
                        continue
                    
                    kp = keypoints[idx]
                    kp_normalized = self.normalize_keypoints(kp, w, h)
                    
                    # Update tracker
                    self.tracker.update(person_id, kp_normalized)
                    tracked_count += 1
                    self.total_detections += 1
                    
                    # Draw skeleton
                    self.draw_skeleton(frame, kp)
                    
                    # Classify
                    behavior_name = None
                    confidence = 0.0
                    
                    if self.tracker.is_ready(person_id):
                        sequence = self.tracker.get_sequence(person_id)
                        predicted_class, conf, probs = self.classify_sequence(sequence)
                        
                        if predicted_class is not None:
                            self.tracker.add_prediction(person_id, predicted_class)
                            smoothed_class = self.tracker.get_smoothed_prediction(person_id)
                            
                            if smoothed_class is not None:
                                behavior_name = self.class_names[smoothed_class]
                                confidence = conf
                                self.behavior_counts[behavior_name] += 1
                    
                    # Draw box
                    color = self.colors[person_id % len(self.colors)]
                    self.draw_person_box(frame, bbox, person_id, behavior_name, confidence, color)
        
        # Cleanup
        self.tracker.cleanup()
        
        # Draw info panel
        frame = self.draw_info_panel(frame, tracked_count)
        
        self.frame_count += 1
        return frame
    
    def process_video(self, video_path, output_path=None, use_notebook=True):
        """
        Process video file.
        
        Args:
            video_path: Input video path
            output_path: Output path (default: input_name_annotated.mp4)
            use_notebook: Use notebook progress bar
        """
        video_path = Path(video_path)
        
        if not video_path.exists():
            print(f"❌ Error: Video not found: {video_path}")
            return None
        
        # Determine output path
        if output_path is None:
            output_path = video_path.parent / f"{video_path.stem}_annotated{video_path.suffix}"
        else:
            output_path = Path(output_path)
        
        print(f"\n🎬 Processing: {video_path.name}")
        print(f"📁 Output: {output_path}")
        
        # Open video
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Cannot open video")
            return None
        
        # Video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Frames: {total_frames}")
        
        # Video writer
        fourcc = cv2.VideoWriter_fourcc(*'mp4v')
        writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
        
        # Progress bar
        if use_notebook:
            try:
                pbar = notebook_tqdm(total=total_frames, desc="Processing")
            except:
                pbar = console_tqdm(total=total_frames, desc="Processing")
        else:
            pbar = console_tqdm(total=total_frames, desc="Processing")
        
        # Process frames
        start_time = time.time()
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process
                annotated_frame = self.process_frame(frame)
                
                # Write
                writer.write(annotated_frame)
                
                # Update progress
                pbar.update(1)
        
        finally:
            pbar.close()
            cap.release()
            writer.release()
        
        # Statistics
        elapsed = time.time() - start_time
        avg_fps = total_frames / elapsed
        
        print("\n" + "=" * 60)
        print("PROCESSING COMPLETE")
        print("=" * 60)
        print(f"✓ Output saved: {output_path}")
        print(f"✓ Processed: {total_frames} frames in {elapsed:.1f}s")
        print(f"✓ Average FPS: {avg_fps:.1f}")
        print(f"✓ Total detections: {self.total_detections}")
        
        if self.behavior_counts:
            print(f"\n📊 Detected Behaviors:")
            for behavior, count in sorted(self.behavior_counts.items(), key=lambda x: -x[1]):
                print(f"   {behavior}: {count} instances")
        
        print("=" * 60)
        
        return output_path


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Google Colab Behavior Recognition',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Process video (output in same folder)
    python inference_colab.py --video input.mp4
    
    # Custom output path
    python inference_colab.py --video input.mp4 --output result.mp4
    
    # Use CPU
    python inference_colab.py --video input.mp4 --device cpu
        """
    )
    
    parser.add_argument('--video', type=str, required=True,
                       help='Path to input video file')
    parser.add_argument('--output', type=str, default=None,
                       help='Output path (default: input_name_annotated.mp4 in same folder)')
    parser.add_argument('--model', type=str, default=None,
                       help='LSTM model path (default: models/lstm_mn/best_model.pth)')
    parser.add_argument('--pose-model', type=str, default=None,
                       help='YOLOv8-pose path (default: models/YOLOv8/yolov8m-pose.pt)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device (default: cuda)')
    parser.add_argument('--no-notebook', action='store_true',
                       help='Disable notebook-style progress bar')
    
    args = parser.parse_args()
    
    # Paths
    model_path = args.model if args.model else config.LSTM_MODEL_DIR / "best_model.pth"
    pose_model_path = args.pose_model if args.pose_model else config.YOLO_POSE_MODEL
    
    # Validate
    if not Path(model_path).exists():
        print(f"❌ LSTM model not found: {model_path}")
        return
    
    if not Path(pose_model_path).exists():
        print(f"❌ YOLOv8-pose not found: {pose_model_path}")
        return
    
    # Initialize
    inference = ColabBehaviorInference(model_path, pose_model_path, device=args.device)
    
    # Process
    output = inference.process_video(
        args.video,
        output_path=args.output,
        use_notebook=not args.no_notebook
    )
    
    if output:
        print(f"\n✅ Success! Download: {output}")


if __name__ == "__main__":
    main()
