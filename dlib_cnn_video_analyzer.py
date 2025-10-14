"""
Dlib CNN Video Analyzer with Live Display and Post-Analysis Graphs
Combines real-time video processing with FaceNet mechanics and post-processing graphs
"""

import os
import sys
import cv2
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import json
import time
import torch
from datetime import datetime
from collections import defaultdict, deque
from pathlib import Path
from ultralytics import YOLO
import glob

# Add project root with debugging
project_root = os.path.abspath(os.path.dirname(__file__))
dlib_cnn_path = os.path.join(project_root, "face_recognition", "Dlibs CNN")

print(f"[DEBUG] Current file: {__file__}")
print(f"[DEBUG] Project root: {project_root}")
print(f"[DEBUG] Dlib CNN path: {dlib_cnn_path}")
print(f"[DEBUG] Dlib CNN path exists: {os.path.exists(dlib_cnn_path)}")

if os.path.exists(dlib_cnn_path):
    print(f"[DEBUG] Files in Dlib CNN directory:")
    for f in os.listdir(dlib_cnn_path):
        print(f"  - {f}")

# Add to path
for path in [project_root, dlib_cnn_path]:
    if path not in sys.path:
        sys.path.insert(0, path)
        print(f"[DEBUG] Added to sys.path: {path}")

# Try import
try:
    from dlib_face_recognizer import DlibCNNRecognizer
    print("[INFO] ✅ Successfully imported DlibCNNRecognizer")
except ImportError as e:
    print(f"[ERROR] Failed to import: {e}")
    print(f"[DEBUG] sys.path: {sys.path[:3]}")
    sys.exit(1)

# Set style for graphs
plt.style.use('default')
sns.set_palette("husl")
plt.rcParams['figure.figsize'] = (12, 8)
plt.rcParams['font.size'] = 12

class DlibCNNVideoAnalyzer:
    def __init__(self, video_path, output_dir="model_comparison_results"):
        self.video_path = video_path
        self.video_name = Path(video_path).stem
        
        # Create organized folder structure (matching FaceNet)
        self.base_output_dir = output_dir
        self.dlib_dir = os.path.join(output_dir, "DlibCNN")
        self.graphs_dir = os.path.join(self.dlib_dir, "graphs")
        self.data_dir = os.path.join(self.dlib_dir, "data")
        self.comparisons_dir = os.path.join(output_dir, "Comparisons")
        
        # Ensure directories exist
        for d in [self.dlib_dir, self.graphs_dir, self.data_dir, self.comparisons_dir]:
            os.makedirs(d, exist_ok=True)
        
        self.results = {
            'frames': [],
            'recognitions': []
        }
        
        # Tracking settings (matching FaceNet main)
        self.BACK_VIEW_TOLERANCE_FRAMES = 600
        self.FACE_RECOG_EVERY_N_FRAMES = 30
        self.IDENTITY_CONFIDENCE_DECAY = 0.995
        self.MIN_IDENTITY_CONFIDENCE = 0.15
        
        # Tracking data structures
        self.track_identities = {}
        self.track_face_history = defaultdict(lambda: deque(maxlen=300))
        self.track_last_face_frame = {}
        self.FACE_RECOG_COOLDOWN_PER_TRACK = {}
        
        print(f"[INFO] Dlib CNN Video Analyzer initialized")
        print(f"[INFO] Video: {video_path}")
        print(f"[INFO] Output: {self.dlib_dir}")
    
    def detect_person_pose_from_body(self, person_crop):
        """Detect pose from body (simplified)"""
        try:
            if person_crop is None or person_crop.size == 0:
                return "unknown", 0.0
            
            h, w = person_crop.shape[:2]
            if h < 80 or w < 40:
                return "frontal", 0.5
            
            gray = cv2.cvtColor(person_crop, cv2.COLOR_BGR2GRAY)
            head_region = gray[:h//3, :]
            
            # Simple back view detection
            head_edges = cv2.Canny(head_region, 30, 100)
            head_edge_density = np.count_nonzero(head_edges) / max(head_edges.size, 1)
            
            # Symmetry check
            left_half = gray[:, :w//2]
            right_half = cv2.flip(gray[:, w//2:], 1)
            min_w = min(left_half.shape[1], right_half.shape[1])
            
            if min_w > 0:
                left_half = left_half[:, :min_w]
                right_half = right_half[:, :min_w]
                
                if left_half.shape == right_half.shape:
                    diff = cv2.absdiff(left_half, right_half)
                    symmetry_score = 1.0 - (np.mean(diff) / 255.0)
                else:
                    symmetry_score = 0.5
            else:
                symmetry_score = 0.5
            
            back_score = 0.0
            if head_edge_density < 0.06:
                back_score += 0.4
            if symmetry_score > 0.7:
                back_score += 0.3
            
            if back_score > 0.6:
                return "back_view", back_score
            elif back_score > 0.4:
                return "partial_back", back_score
            else:
                return "frontal", 1.0 - back_score
                
        except Exception:
            return "frontal", 0.5
    
    def update_track_identity(self, track_id, face_result, person_crop, frame_num):
        """Update tracking identity with pose detection"""
        
        name = face_result['name']
        conf = face_result['confidence']
        
        pose, pose_confidence = self.detect_person_pose_from_body(person_crop)
        
        if track_id not in self.track_identities:
            self.track_identities[track_id] = {
                'name': name,
                'confidence': conf,
                'last_face_frame': frame_num if name != 'Unknown' else -1,
                'stable': False,
                'pose_history': deque(maxlen=30),
                'consecutive_back_frames': 0,
                'identity_locked': False,
                'frames_since_face_lost': 0,
                'total_face_detections': 0
            }
        
        identity = self.track_identities[track_id]
        identity['pose_history'].append((pose, pose_confidence, frame_num))
        
        self.track_face_history[track_id].append({
            'name': name,
            'confidence': conf,
            'frame': frame_num,
            'pose': pose
        })
        
        if name != 'Unknown':
            identity['total_face_detections'] += 1
            
            if conf > 0.7:
                identity['name'] = name
                identity['confidence'] = conf
                identity['stable'] = True
                identity['last_face_frame'] = frame_num
                identity['consecutive_back_frames'] = 0
                identity['frames_since_face_lost'] = 0
                identity['identity_locked'] = True
        else:
            identity['frames_since_face_lost'] += 1
            
            if pose in ['back_view', 'partial_back']:
                identity['consecutive_back_frames'] += 1
                
                if identity['identity_locked']:
                    identity['confidence'] *= 0.9995
                else:
                    identity['confidence'] *= 0.985
            else:
                if identity['identity_locked']:
                    identity['confidence'] *= 0.998
                else:
                    identity['confidence'] *= self.IDENTITY_CONFIDENCE_DECAY
        
        if identity['confidence'] < self.MIN_IDENTITY_CONFIDENCE:
            identity['stable'] = False
    
    def get_consensus_identity(self, track_id, frames_since_face):
        """Get consensus identity"""
        if track_id not in self.track_identities:
            return 'Unknown', 0.0
        
        identity = self.track_identities[track_id]
        
        if not identity['stable']:
            return 'Unknown', 0.0
        
        return identity['name'], identity['confidence']
    
    def recognize_face_in_crop_optimized(self, person_crop, original_frame, person_bbox):
        """Optimized face recognition using Dlib CNN with preprocessing (copied from main)"""
        if person_crop is None or person_crop.size == 0:
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': 0.0}
        
        try:
            # Apply preprocessing
            processed_crop = person_crop.copy()
            
            # Gamma correction
            gray_check = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2GRAY)
            mean_brightness = np.mean(gray_check)
            
            if mean_brightness < 80:  # Dark image
                # Auto gamma correction
                gamma = 100.0 / max(mean_brightness, 1.0)
                gamma = max(min(gamma, 1.8), 0.6)
                invGamma = 1.0 / gamma
                table = np.array([((i / 255.0) ** invGamma) * 255 for i in np.arange(256)]).astype('uint8')
                processed_crop = cv2.LUT(processed_crop, table)
            
            # CLAHE enhancement
            person_rgb = cv2.cvtColor(processed_crop, cv2.COLOR_BGR2RGB)
            lab = cv2.cvtColor(person_rgb, cv2.COLOR_RGB2LAB)
            l, a, b = cv2.split(lab)
            clahe = cv2.createCLAHE(clipLimit=2.0, tileGridSize=(8, 8))
            cl = clahe.apply(l)
            limg = cv2.merge((cl, a, b))
            processed_rgb = cv2.cvtColor(limg, cv2.COLOR_LAB2RGB)
            processed_crop = cv2.cvtColor(processed_rgb, cv2.COLOR_RGB2BGR)
            
            # Use the recognizer's all-in-one method
            result = self.recognizer.recognize_face_in_crop(processed_crop, original_frame, person_bbox)
            return result
            
        except Exception as e:
            print(f"[ERROR] Optimized face recognition failed: {e}")
            return {'name': 'Unknown', 'confidence': 0.0, 'face_bbox': None, 'inference_time': 0.0}
    
    def process_video_with_analysis(self, show_video=True):
        """Process video with live display and tracking"""
        
        print(f"[INFO] Initializing Dlib CNN models...")
        
        # Initialize models
        self.recognizer = DlibCNNRecognizer()
        self.yolo = YOLO("models/YOLOv8/yolov8n.pt")
        
        cap = cv2.VideoCapture(self.video_path)
        if not cap.isOpened():
            print(f"[ERROR] Cannot open video: {self.video_path}")
            return False
        
        fps = cap.get(cv2.CAP_PROP_FPS)
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        
        print(f"[INFO] Video: {width}x{height} @ {fps}fps, {total_frames} frames")
        print(f"[INFO] Processing with Dlib CNN recognition...")
        
        frame_count = 0
        all_recognitions = []
        
        PROCESS_EVERY_N = 5
        PERSON_CONF_THRESHOLD = 0.6
        
        colors = [
            (255, 0, 0), (0, 255, 0), (0, 0, 255), (255, 255, 0), 
            (255, 0, 255), (0, 255, 255), (128, 0, 128), (255, 165, 0)
        ]
        
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            frame_start = time.time()
            
            if frame_count % PROCESS_EVERY_N != 0:
                frame_count += 1
                continue
            
            display_frame = frame.copy()
            
            try:
                # YOLO tracking
                results = self.yolo.track(
                    frame,
                    persist=True,
                    tracker="bytetrack.yaml",
                    classes=[0],
                    conf=PERSON_CONF_THRESHOLD,
                    verbose=False
                )
                
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        bboxes = boxes.xyxy.cpu().numpy()
                        confidences = boxes.conf.cpu().numpy()
                        
                        for track_id, bbox, conf in zip(track_ids, bboxes, confidences):
                            x1, y1, x2, y2 = bbox.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                            
                            person_bbox = (x1, y1, x2, y2)
                            person_crop = frame[y1:y2, x1:x2]
                            
                            # Face recognition with cooldown
                            should_recognize = (
                                track_id not in self.FACE_RECOG_COOLDOWN_PER_TRACK or
                                frame_count - self.FACE_RECOG_COOLDOWN_PER_TRACK[track_id] >= self.FACE_RECOG_EVERY_N_FRAMES
                            )
                            
                            if should_recognize:
                                self.FACE_RECOG_COOLDOWN_PER_TRACK[track_id] = frame_count
                                
                                inference_start = time.time()
                                
                                # FIX: Use the optimized method with preprocessing
                                face_result = self.recognize_face_in_crop_optimized(
                                    person_crop, frame, person_bbox
                                )
                                
                                # Add inference time
                                face_result['inference_time'] = (time.time() - inference_start) * 1000
                                
                                self.track_last_face_frame[track_id] = frame_count
                            else:
                                face_result = {
                                    'name': 'Unknown',
                                    'confidence': 0.0,
                                    'inference_time': 0.0,
                                    'face_bbox': None
                                }
                            
                            # Update tracking
                            self.update_track_identity(track_id, face_result, person_crop, frame_count)
                            
                            frames_since_face = frame_count - self.track_last_face_frame.get(track_id, frame_count)
                            identity_name, identity_conf = self.get_consensus_identity(track_id, frames_since_face)
                            
                            pose, pose_conf = self.detect_person_pose_from_body(person_crop)
                            identity = self.track_identities.get(track_id, {})
                            
                            # Store data
                            recognition_data = {
                                'frame': frame_count,
                                'track_id': track_id,
                                'name': identity_name,
                                'confidence': identity_conf,
                                'inference_time_ms': face_result['inference_time'],
                                'pose': pose,
                                'pose_confidence': pose_conf,
                                'is_locked': identity.get('identity_locked', False),
                                'consecutive_back_frames': identity.get('consecutive_back_frames', 0)
                            }
                            all_recognitions.append(recognition_data)
                            
                            # Display
                            color = (0, 255, 0) if identity_name != 'Unknown' else (0, 0, 255)
                            thickness = 3 if identity.get('identity_locked', False) else 2
                            
                            cv2.rectangle(display_frame, (x1, y1), (x2, y2), color, thickness)
                            
                            pose_info = f" [{pose[:4].upper()}]" if pose in ['back_view', 'partial_back'] else ""
                            lock_info = " 🔒" if identity.get('identity_locked', False) else ""
                            
                            label = f"ID:{track_id} {identity_name}"
                            if identity_conf > 0:
                                label += f" ({identity_conf:.2f})"
                            label += pose_info + lock_info
                            
                            label_y = max(30, y1 - 10)
                            (label_w, label_h), _ = cv2.getTextSize(label, cv2.FONT_HERSHEY_SIMPLEX, 0.6, 2)
                            cv2.rectangle(display_frame, (x1, label_y - label_h - 5), 
                                        (x1 + label_w + 5, label_y + 5), (0, 0, 0), -1)
                            cv2.putText(display_frame, label, (x1 + 2, label_y), 
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # FPS info
                frame_time = (time.time() - frame_start) * 1000
                fps_actual = 1000.0 / max(frame_time, 1)
                locked_tracks = sum(1 for t in self.track_identities.values() if t.get('identity_locked', False))
                
                info = f"Frame: {frame_count}/{total_frames} | FPS: {fps_actual:.1f} | Locked: {locked_tracks}"
                cv2.putText(display_frame, info, (10, 30), 
                          cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
                
                if frame_count % 100 == 0:
                    progress = (frame_count / total_frames) * 100
                    print(f"  Progress: {progress:.1f}% - Locked: {locked_tracks}")
                
            except Exception as e:
                print(f"[ERROR] Frame {frame_count}: {e}")
            
            if show_video:
                cv2.imshow('Dlib CNN Video Analysis', display_frame)
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    print("\n[INFO] User quit")
                    break
            
            frame_count += 1
        
        cap.release()
        if show_video:
            cv2.destroyAllWindows()
        
        self.results['recognitions'] = all_recognitions
        
        print(f"\n[INFO] Processing complete!")
        print(f"  Total recognitions: {len(all_recognitions)}")
        
        return True
    
    def calculate_simple_metrics(self):
        """Calculate metrics (matching FaceNet)"""
        
        if not self.results['recognitions']:
            print("[ERROR] No data to analyze")
            return None
        
        df = pd.DataFrame(self.results['recognitions'])
        
        known_faces = df[df['name'] != 'Unknown']
        avg_confidence = known_faces['confidence'].mean() if len(known_faces) > 0 else 0.0
        avg_inference_time = df['inference_time_ms'].mean()
        avg_accuracy = (len(known_faces) / len(df)) if len(df) > 0 else 0.0
        
        back_view_detections = len(df[df['pose'] == 'back_view'])
        locked_detections = len(df[df['is_locked'] == True])
        
        stats = {
            'model_name': 'DlibCNN',
            'video_name': self.video_name,
            'avg_confidence': avg_confidence,
            'avg_inference_time_ms': avg_inference_time,
            'avg_accuracy': avg_accuracy,
            'total_detections': len(df),
            'successful_recognitions': len(known_faces),
            'recognition_rate': avg_accuracy,
            'back_view_detections': back_view_detections,
            'back_view_percentage': (back_view_detections / len(df)) * 100 if len(df) > 0 else 0,
            'locked_detections': locked_detections,
            'locked_percentage': (locked_detections / len(df)) * 100 if len(df) > 0 else 0,
            'unique_tracks': len(self.track_identities),
            'locked_tracks': sum(1 for t in self.track_identities.values() if t.get('identity_locked', False))
        }
        
        return stats
    
    def create_simple_graph(self, stats):
        """Create performance graph (matching FaceNet style)"""
        
        print(f"\n[INFO] Creating Dlib CNN performance graph...")
        
        fig, ax = plt.subplots(1, 1, figsize=(12, 8))
        
        metrics = ['Average\nConfidence', 'Average Inference\nTime (ms)', 'Average\nAccuracy']
        values = [stats['avg_confidence'], stats['avg_inference_time_ms'], stats['avg_accuracy']]
        colors = ['#FF6B6B', '#4ECDC4', '#45B7D1']
        
        bars = ax.bar(metrics, values, color=colors, alpha=0.8, edgecolor='black', linewidth=1.5)
        
        ax.set_title(f'Dlib CNN Performance Metrics - {self.video_name}', 
                    fontsize=16, fontweight='bold', pad=20)
        ax.set_ylabel('Value', fontsize=14, fontweight='bold')
        ax.grid(True, alpha=0.3, axis='y')
        
        for i, (bar, value) in enumerate(zip(bars, values)):
            height = bar.get_height()
            if i == 0:
                label_text = f'{value:.3f}'
            elif i == 1:
                label_text = f'{value:.1f}ms'
            else:
                label_text = f'{value:.1%}'
            
            ax.text(bar.get_x() + bar.get_width()/2., height + max(values) * 0.01,
                   label_text, ha='center', va='bottom', fontsize=12, fontweight='bold')
        
        ax.set_ylim(0, max(values) * 1.15)
        
        summary_text = f"""
📊 DLIB CNN SUMMARY

🎯 METRICS:
• Confidence: {stats['avg_confidence']:.3f}
• Inference: {stats['avg_inference_time_ms']:.1f}ms
• Accuracy: {stats['avg_accuracy']:.1%}

📈 STATS:
• Total: {stats['total_detections']:,}
• Success: {stats['successful_recognitions']:,}
• Rate: {stats['recognition_rate']:.1%}

🔄 TRACKING:
• Back View: {stats['back_view_percentage']:.1f}%
• Locked: {stats['locked_percentage']:.1f}%
• Tracks: {stats['unique_tracks']:,}
        """
        
        ax.text(0.02, 0.98, summary_text.strip(), transform=ax.transAxes,
               fontsize=9, verticalalignment='top', fontfamily='monospace',
               bbox=dict(boxstyle='round,pad=0.5', facecolor='lightcyan', alpha=0.8))
        
        plt.tight_layout()
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        graph_path = os.path.join(self.graphs_dir, f"dlib_cnn_performance_{self.video_name}_{timestamp}.png")
        plt.savefig(graph_path, dpi=300, bbox_inches='tight')
        
        standard_graph_path = os.path.join(self.graphs_dir, f"dlib_cnn_metrics_{self.video_name}.png")
        plt.savefig(standard_graph_path, dpi=300, bbox_inches='tight')
        
        print(f"[INFO] ✅ Saved graphs: {graph_path}")
        
        plt.show()
        
        return graph_path, standard_graph_path
    
    def save_results(self, stats):
        """Save results (matching FaceNet)"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        if self.results['recognitions']:
            csv_path = os.path.join(self.data_dir, f"dlib_cnn_data_{self.video_name}.csv")
            df = pd.DataFrame(self.results['recognitions'])
            df.to_csv(csv_path, index=False)
            print(f"[INFO] ✅ Saved data: {csv_path}")
        
        stats_path = os.path.join(self.data_dir, f"dlib_cnn_stats_{self.video_name}.json")
        with open(stats_path, 'w') as f:
            json.dump(stats, f, indent=2)
        print(f"[INFO] ✅ Saved stats: {stats_path}")
        
        comparison_summary = {
            'model': 'DlibCNN',
            'video': self.video_name,
            'metrics': {
                'avg_confidence': stats['avg_confidence'],
                'avg_inference_time_ms': stats['avg_inference_time_ms'],
                'avg_accuracy': stats['avg_accuracy']
            },
            'timestamp': timestamp
        }
        
        comparison_path = os.path.join(self.comparisons_dir, f"dlib_cnn_summary_{self.video_name}.json")
        with open(comparison_path, 'w') as f:
            json.dump(comparison_summary, f, indent=2)
        print(f"[INFO] ✅ Saved comparison: {comparison_path}")
        
        return True

def main():
    parser = argparse.ArgumentParser(description="Dlib CNN video analysis with live display and graphs")
    parser.add_argument("--video", "-v",
                       default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST3.mp4",
                       help="Path to video file")
    parser.add_argument("--output", "-o",
                       default="model_comparison_results",
                       help="Output directory")
    parser.add_argument("--no-display", "-n",
                       action="store_true",
                       help="Run without video display")
    
    args = parser.parse_args()
    
    if not os.path.exists(args.video):
        print(f"[ERROR] Video not found: {args.video}")
        return
    
    print("🎬 DLIB CNN VIDEO ANALYZER")
    print("=" * 50)
    
    analyzer = DlibCNNVideoAnalyzer(args.video, args.output)
    
    if not analyzer.process_video_with_analysis(show_video=not args.no_display):
        print("[ERROR] Processing failed")
        return
    
    stats = analyzer.calculate_simple_metrics()
    if stats is None:
        print("[ERROR] Metrics calculation failed")
        return
    
    analyzer.create_simple_graph(stats)
    analyzer.save_results(stats)
    
    print(f"\n✅ Dlib CNN analysis complete!")
    print(f"📁 Results: {analyzer.dlib_dir}")

if __name__ == "__main__":
    main()