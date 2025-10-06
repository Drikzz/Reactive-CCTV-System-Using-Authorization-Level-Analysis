"""
Real-time Behavior Recognition Inference
Uses YOLOv8 tracking + LSTM classifier for behavior classification from pose sequences.

Usage:
    # Webcam
    python inference_behavior.py --webcam
    
    # Video file
    python inference_behavior.py --video path/to/video.mp4
    
    # Save output
    python inference_behavior.py --video input.mp4 --output runs/inference/output.mp4
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from collections import deque, defaultdict
from ultralytics import YOLO
import argparse
import time
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
        
        # Timeout for removing inactive tracks (frames)
        self.track_timeout = 30
    
    def update(self, person_id, keypoints):
        """
        Update sequence for a tracked person.
        
        Args:
            person_id: Unique tracking ID
            keypoints: [17, 2] normalized keypoints
        """
        self.person_sequences[person_id].append(keypoints)
        self.person_last_seen[person_id] = 0
    
    def add_prediction(self, person_id, predicted_class):
        """Add prediction to smoothing buffer."""
        self.person_predictions[person_id].append(predicted_class)
    
    def get_sequence(self, person_id):
        """Get current sequence for person."""
        return list(self.person_sequences[person_id])
    
    def get_smoothed_prediction(self, person_id):
        """Get majority vote prediction from smoothing window."""
        if person_id not in self.person_predictions or len(self.person_predictions[person_id]) == 0:
            return None
        
        # Majority vote
        predictions = list(self.person_predictions[person_id])
        return max(set(predictions), key=predictions.count)
    
    def is_ready(self, person_id):
        """Check if person has enough frames for classification."""
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


class BehaviorInference:
    """Real-time behavior classification with person tracking."""
    
    def __init__(self, model_path, pose_model_path, device='cuda'):
        """
        Initialize inference system.
        
        Args:
            model_path: Path to trained LSTM checkpoint
            pose_model_path: Path to YOLOv8-pose model
            device: 'cuda' or 'cpu'
        """
        self.device = torch.device(device if torch.cuda.is_available() else 'cpu')
        
        print("=" * 60)
        print("BEHAVIOR RECOGNITION INFERENCE")
        print("=" * 60)
        
        # Load LSTM classifier
        print(f"\n🧠 Loading LSTM classifier from {model_path}...")
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
        
        print(f"✓ LSTM model loaded")
        print(f"   Classes: {self.class_names}")
        print(f"   Device: {self.device}")
        
        # Load YOLOv8-pose model with tracking
        print(f"\n📦 Loading YOLOv8-pose model from {pose_model_path}...")
        self.pose_model = YOLO(str(pose_model_path))
        self.pose_model.to(self.device)
        print("✓ YOLOv8-pose model loaded (tracking enabled)")
        
        # Initialize person tracker
        self.sequence_length = config.FIXED_SEQUENCE_LENGTH or 60
        self.tracker = PersonTracker(sequence_length=self.sequence_length, smoothing_window=5)
        
        # Statistics
        self.frame_count = 0
        self.fps = 0
        self.last_time = time.time()
        
        # Colors for visualization
        self.colors = self._generate_colors(20)  # Support up to 20 tracked persons
    
    def _generate_colors(self, n):
        """Generate distinct colors for tracking visualization."""
        colors = []
        for i in range(n):
            hue = int(180 * i / n)
            color = cv2.cvtColor(np.uint8([[[hue, 255, 255]]]), cv2.COLOR_HSV2BGR)[0][0]
            colors.append(tuple(map(int, color)))
        return colors
    
    def normalize_keypoints(self, keypoints, frame_width, frame_height):
        """
        Normalize keypoints to [0, 1] range.
        
        Args:
            keypoints: [17, 2] keypoints in pixel coordinates
            frame_width: Frame width
            frame_height: Frame height
            
        Returns:
            Normalized keypoints [17, 2]
        """
        kp = keypoints.copy()
        kp[:, 0] = kp[:, 0] / frame_width
        kp[:, 1] = kp[:, 1] / frame_height
        return kp
    
    def classify_sequence(self, sequence):
        """
        Classify a pose sequence.
        
        Args:
            sequence: List of [17, 2] keypoint arrays
            
        Returns:
            tuple: (predicted_class, confidence, probabilities)
        """
        if len(sequence) < self.sequence_length:
            return None, 0.0, None
        
        # Convert to tensor [seq_len, 17, 2] -> [seq_len, 34]
        seq_array = np.array(sequence[-self.sequence_length:])  # Take last N frames
        seq_array = seq_array.reshape(seq_array.shape[0], -1)  # Flatten to [seq_len, 34]
        
        seq_tensor = torch.tensor(seq_array, dtype=torch.float32).unsqueeze(0)  # [1, seq_len, 34]
        seq_tensor = seq_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.lstm_model(seq_tensor)
            probabilities = torch.softmax(outputs, dim=1)
            confidence, predicted = probabilities.max(1)
        
        predicted_class = predicted.item()
        confidence_value = confidence.item()
        all_probs = probabilities.cpu().numpy()[0]
        
        return predicted_class, confidence_value, all_probs
    
    def draw_skeleton(self, frame, keypoints):
        """
        Draw pose skeleton on frame.
        
        Args:
            frame: BGR image
            keypoints: [17, 2] keypoints in pixel coordinates
        """
        kp = keypoints.astype(int)
        
        # COCO pose skeleton connections
        skeleton = [
            [15, 13], [13, 11], [16, 14], [14, 12], [11, 12],  # Legs
            [5, 11], [6, 12], [5, 6],  # Torso
            [5, 7], [6, 8], [7, 9], [8, 10],  # Arms
            [0, 1], [0, 2], [1, 3], [2, 4]  # Head
        ]
        
        # Draw connections
        for connection in skeleton:
            pt1 = tuple(kp[connection[0]])
            pt2 = tuple(kp[connection[1]])
            cv2.line(frame, pt1, pt2, (0, 255, 0), 2)
        
        # Draw keypoints
        for point in kp:
            cv2.circle(frame, tuple(point), 3, (0, 0, 255), -1)
    
    def draw_person_info(self, frame, bbox, person_id, behavior, confidence, color):
        """
        Draw bounding box and behavior label for a person.
        
        Args:
            frame: BGR image
            bbox: [x1, y1, x2, y2] bounding box
            person_id: Tracking ID
            behavior: Predicted behavior class name
            confidence: Prediction confidence
            color: Color for this person
        """
        x1, y1, x2, y2 = bbox.astype(int)
        
        # Draw bounding box
        cv2.rectangle(frame, (x1, y1), (x2, y2), color, 2)
        
        # Prepare label
        if behavior is not None:
            label = f"ID:{person_id} | {behavior} ({confidence*100:.1f}%)"
            label_color = (0, 255, 0) if confidence > 0.7 else (0, 165, 255)
        else:
            label = f"ID:{person_id} | Collecting..."
            label_color = (150, 150, 150)
        
        # Draw label background
        (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
        cv2.rectangle(frame, (x1, y1 - label_h - 10), (x1 + label_w + 10, y1), color, -1)
        
        # Draw label text
        cv2.putText(frame, label, (x1 + 5, y1 - 5),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
    
    def draw_stats(self, frame):
        """Draw FPS and tracking statistics."""
        h, w = frame.shape[:2]
        
        # Semi-transparent background
        overlay = frame.copy()
        cv2.rectangle(overlay, (10, 10), (300, 100), (0, 0, 0), -1)
        frame = cv2.addWeighted(overlay, 0.6, frame, 0.4, 0)
        
        # FPS
        cv2.putText(frame, f"FPS: {self.fps:.1f}", (20, 35),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        # Active tracks
        active_tracks = len(self.tracker.person_sequences)
        cv2.putText(frame, f"Tracked Persons: {active_tracks}", (20, 65),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (200, 200, 200), 2)
        
        # Sequence length
        cv2.putText(frame, f"Window: {self.sequence_length} frames", (20, 90),
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (150, 150, 150), 1)
        
        return frame
    
    def process_frame(self, frame):
        """
        Process a single frame with tracking and classification.
        
        Args:
            frame: BGR image from OpenCV
            
        Returns:
            Annotated frame
        """
        h, w = frame.shape[:2]
        
        # Run YOLOv8 tracking (pose + tracking)
        results = self.pose_model.track(
            frame,
            conf=config.YOLO_CONF_THRESHOLD,
            persist=True,  # Enable tracking
            verbose=False
        )
        
        # Process each tracked person
        if results[0].boxes is not None and results[0].boxes.id is not None:
            boxes = results[0].boxes.xyxy.cpu().numpy()
            track_ids = results[0].boxes.id.cpu().numpy().astype(int)
            
            if results[0].keypoints is not None:
                keypoints = results[0].keypoints.xy.cpu().numpy()  # [num_persons, 17, 2]
                
                for idx, (bbox, person_id) in enumerate(zip(boxes, track_ids)):
                    if idx >= len(keypoints):
                        continue
                    
                    kp = keypoints[idx]  # [17, 2]
                    
                    # Normalize keypoints
                    kp_normalized = self.normalize_keypoints(kp, w, h)
                    
                    # Update tracker
                    self.tracker.update(person_id, kp_normalized)
                    
                    # Draw skeleton
                    self.draw_skeleton(frame, kp)
                    
                    # Classify if enough frames collected
                    behavior_name = None
                    confidence = 0.0
                    
                    if self.tracker.is_ready(person_id):
                        sequence = self.tracker.get_sequence(person_id)
                        predicted_class, conf, probs = self.classify_sequence(sequence)
                        
                        if predicted_class is not None:
                            # Add to smoothing buffer
                            self.tracker.add_prediction(person_id, predicted_class)
                            
                            # Get smoothed prediction
                            smoothed_class = self.tracker.get_smoothed_prediction(person_id)
                            if smoothed_class is not None:
                                behavior_name = self.class_names[smoothed_class]
                                confidence = conf
                    
                    # Draw person info
                    color = self.colors[person_id % len(self.colors)]
                    self.draw_person_info(frame, bbox, person_id, behavior_name, confidence, color)
        
        # Cleanup inactive tracks
        self.tracker.cleanup()
        
        # Draw statistics
        frame = self.draw_stats(frame)
        
        # Update FPS
        self.frame_count += 1
        if self.frame_count % 10 == 0:
            current_time = time.time()
            self.fps = 10 / (current_time - self.last_time)
            self.last_time = current_time
        
        return frame
    
    def run_webcam(self, camera_id=0):
        """
        Run inference on webcam feed.
        
        Args:
            camera_id: Camera device ID
        """
        print(f"\n📹 Starting webcam inference (Camera {camera_id})...")
        print("Press 'q' to quit, 's' to save screenshot")
        
        cap = cv2.VideoCapture(camera_id)
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open camera {camera_id}")
            return
        
        # Set camera properties
        cap.set(cv2.CAP_PROP_FRAME_WIDTH, 1280)
        cap.set(cv2.CAP_PROP_FRAME_HEIGHT, 720)
        cap.set(cv2.CAP_PROP_FPS, 30)
        
        screenshot_count = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    print("❌ Error reading frame")
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Display
                cv2.imshow('Behavior Recognition - Webcam', annotated_frame)
                
                # Handle keys
                key = cv2.waitKey(1) & 0xFF
                if key == ord('q'):
                    break
                elif key == ord('s'):
                    screenshot_path = f"screenshot_{screenshot_count:04d}.jpg"
                    cv2.imwrite(screenshot_path, annotated_frame)
                    print(f"📸 Screenshot saved: {screenshot_path}")
                    screenshot_count += 1
        
        finally:
            cap.release()
            cv2.destroyAllWindows()
            print("\n✓ Webcam inference stopped")
    
    def run_video(self, video_path, output_path=None):
        """
        Run inference on video file.
        
        Args:
            video_path: Path to input video
            output_path: Path to save output video (optional)
        """
        print(f"\n🎬 Processing video: {video_path}")
        
        cap = cv2.VideoCapture(str(video_path))
        
        if not cap.isOpened():
            print(f"❌ Error: Cannot open video {video_path}")
            return
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        
        print(f"   Resolution: {width}x{height}")
        print(f"   FPS: {fps}")
        print(f"   Total frames: {total_frames}")
        
        # Setup video writer
        writer = None
        if output_path:
            output_path = Path(output_path)
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            writer = cv2.VideoWriter(str(output_path), fourcc, fps, (width, height))
            print(f"   Output: {output_path}")
        
        print("\nProcessing... Press 'q' to quit")
        
        from tqdm import tqdm
        pbar = tqdm(total=total_frames, desc="Progress")
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Process frame
                annotated_frame = self.process_frame(frame)
                
                # Write to output
                if writer:
                    writer.write(annotated_frame)
                
                # Display
                cv2.imshow('Behavior Recognition - Video', annotated_frame)
                
                # Update progress
                pbar.update(1)
                
                # Handle quit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break
        
        finally:
            pbar.close()
            cap.release()
            if writer:
                writer.release()
            cv2.destroyAllWindows()
            
            print("\n✓ Video processing complete")
            if output_path:
                print(f"✓ Output saved to: {output_path}")


def main():
    """Main entry point."""
    parser = argparse.ArgumentParser(
        description='Real-time Behavior Recognition with Person Tracking',
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
    # Webcam inference
    python inference_behavior.py --webcam
    
    # Video file inference
    python inference_behavior.py --video path/to/video.mp4
    
    # Video with output saving
    python inference_behavior.py --video input.mp4 --output runs/inference/output.mp4
    
    # Use CPU
    python inference_behavior.py --webcam --device cpu
        """
    )
    
    # Input source
    source_group = parser.add_mutually_exclusive_group(required=True)
    source_group.add_argument('--webcam', action='store_true',
                             help='Use webcam as input')
    source_group.add_argument('--video', type=str,
                             help='Path to input video file')
    
    # Optional arguments
    parser.add_argument('--camera-id', type=int, default=0,
                       help='Camera device ID for webcam mode (default: 0)')
    parser.add_argument('--output', type=str, default=None,
                       help='Path to save output video (only for video mode)')
    parser.add_argument('--model', type=str, default=None,
                       help='Path to trained LSTM model (default: models/lstm_mn/best_model.pth)')
    parser.add_argument('--pose-model', type=str, default=None,
                       help='Path to YOLOv8-pose model (default: models/YOLOv8/yolov8m-pose.pt)')
    parser.add_argument('--device', type=str, default='cuda',
                       choices=['cuda', 'cpu'],
                       help='Device to use for inference (default: cuda)')
    
    args = parser.parse_args()
    
    # Determine paths
    model_path = args.model if args.model else config.LSTM_MODEL_DIR / "best_model.pth"
    pose_model_path = args.pose_model if args.pose_model else config.YOLO_POSE_MODEL
    
    # Validate model exists
    if not Path(model_path).exists():
        print(f"❌ Error: LSTM model not found at {model_path}")
        print("Please train the model first using train_lstm_classifier.py")
        return
    
    if not Path(pose_model_path).exists():
        print(f"❌ Error: YOLOv8-pose model not found at {pose_model_path}")
        return
    
    # Initialize inference system
    inference = BehaviorInference(model_path, pose_model_path, device=args.device)
    
    # Run inference
    if args.webcam:
        inference.run_webcam(camera_id=args.camera_id)
    else:
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"❌ Error: Video file not found: {video_path}")
            return
        
        inference.run_video(video_path, output_path=args.output)


if __name__ == "__main__":
    main()
