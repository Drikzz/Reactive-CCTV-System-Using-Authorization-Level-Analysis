"""
Dlib CNN Face Recognition System
Uses YOLOv8 + Dlib's CNN face detector + face recognition
Reuses existing tracker and utilities from the project
"""
import sys
import os
import cv2
import dlib
import numpy as np
from datetime import datetime
from threading import Thread, Event
from queue import Queue, Empty
from collections import deque, defaultdict
from ultralytics import YOLO

# Add project root to path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(__file__)))
sys.path.insert(0, project_root)

# Import existing utilities (fix paths)
from face_recognition.Facenet.facenet_recognition_tracker import RecognitionTracker

# Fix the folder name issue (Dlibs CNN vs DlibsCNN)
sys.path.append(os.path.join(os.path.dirname(__file__)))
from dlib_face_recognizer import DlibCNNRecognizer

# -------------------- CONFIG --------------------
USE_WEBCAM = False
VIDEO_PATH = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST3.mp4"
YOLO_MODEL_PATH = "models/YOLOv8/yolov8n.pt"
DLIB_MODELS_DIR = os.path.join("models", "Dlib")

# Optimize performance settings
RESIZE_WIDTH = 480
PROCESS_EVERY_N = 5
RECOG_THRESHOLD = 0.45
PERSON_CONF_THRESHOLD = 0.6
FACE_RECOG_EVERY_N = 10

# Identity locking settings - Make locking STRONGER
LOCK_CONFIDENCE_THRESHOLD = 0.5       # Lower threshold to lock faster
LOCK_CONSISTENT_FRAMES = 3           # Fewer frames needed to lock (faster locking)
LOCK_TIMEOUT_FRAMES = 30             
LOCK_MIN_CONFIDENCE_TO_MAINTAIN = 0.2  # Even lower threshold to maintain lock
LOCK_BREAK_THRESHOLD = 0.7           # High confidence needed to break an existing lock

SAVE_FACES = True

# Output directories
LOGS_BASE = os.path.join("logs", "DlibCNN")
ANNOTATED_BASE = os.path.join("annotated_frames", "DlibCNN")

# -------------------- MAIN CLASS --------------------
class DlibCNNSystem:
    def __init__(self):
        self.yolo = YOLO(YOLO_MODEL_PATH)
        self.face_recognizer = DlibCNNRecognizer(DLIB_MODELS_DIR)
        self.tracker = RecognitionTracker()
        self.debug_count = 0
        self.last_face_recog_frame = {}
        self.global_identities = {}
        self.identity_persistence_threshold = 0.35
        
        # Identity locking mechanism
        self.track_locks = {}  # {track_id: {'identity': str, 'locked': bool, 'lock_strength': int, 'last_seen': int}}
        self.track_detection_history = defaultdict(lambda: deque(maxlen=LOCK_CONSISTENT_FRAMES * 2))
        
        # Create output directories
        for p in [LOGS_BASE, ANNOTATED_BASE]:
            os.makedirs(p, exist_ok=True)
    
    def update_track_lock_status(self, track_id, detected_identity, confidence, frame_num):
        """Update the locking mechanism for a track - STRONGER LOCKING"""
        
        # Initialize track lock data if needed
        if track_id not in self.track_locks:
            self.track_locks[track_id] = {
                'identity': 'Unknown',
                'locked': False,
                'lock_strength': 0,
                'last_seen': frame_num,
                'lock_start_frame': None,
                'lock_confidence': 0.0,
                'conflicting_detections': 0
            }
        
        track_lock = self.track_locks[track_id]
        track_lock['last_seen'] = frame_num
        
        # Add current detection to history
        self.track_detection_history[track_id].append({
            'identity': detected_identity,
            'confidence': confidence,
            'frame': frame_num
        })
        
        # Debug output for identity switching
        print(f"[DEBUG] Track {track_id}: Detected='{detected_identity}' (conf:{confidence:.3f}), "
              f"Locked='{track_lock.get('identity', 'Unknown')}' (locked:{track_lock['locked']})")
        
        # If already locked, check if we should maintain or break the lock
        if track_lock['locked']:
            locked_identity = track_lock['identity']
            
            # Strong evidence for the locked identity or acceptable low confidence
            if (detected_identity == locked_identity and confidence >= LOCK_MIN_CONFIDENCE_TO_MAINTAIN) or \
               (detected_identity == 'Unknown' and confidence == 0.0):  # No detection is OK when locked
                # Maintain lock - STRENGTHEN IT
                track_lock['lock_strength'] = min(track_lock['lock_strength'] + 2, LOCK_CONSISTENT_FRAMES * 5)
                track_lock['conflicting_detections'] = 0  # Reset conflict counter
                print(f"[LOCK] ✅ Maintaining lock on {locked_identity} for track {track_id} (strength: {track_lock['lock_strength']})")
                return locked_identity, track_lock.get('lock_confidence', confidence)
            
            # Strong evidence against the locked identity - MAKE IT HARDER TO BREAK
            elif detected_identity != 'Unknown' and detected_identity != locked_identity and confidence >= LOCK_BREAK_THRESHOLD:
                track_lock['conflicting_detections'] += 1
                track_lock['lock_strength'] -= 1  # Smaller penalty
                
                print(f"[CONFLICT] ⚠️  Track {track_id}: '{detected_identity}' vs locked '{locked_identity}' "
                      f"(conf:{confidence:.3f}, conflicts:{track_lock['conflicting_detections']}, strength:{track_lock['lock_strength']})")
                
                # Check if we should break the lock - REQUIRE MORE EVIDENCE
                if track_lock['lock_strength'] <= 0 and track_lock['conflicting_detections'] >= 3:
                    print(f"[UNLOCK] 🔓 Breaking lock on {locked_identity} for track {track_id} - conflicting detection: {detected_identity}")
                    track_lock['locked'] = False
                    track_lock['lock_strength'] = 0
                    track_lock['identity'] = 'Unknown'
                    track_lock['conflicting_detections'] = 0
                else:
                    print(f"[LOCK] 🔒 Keeping lock on {locked_identity} despite conflict (strength: {track_lock['lock_strength']})")
                    return locked_identity, track_lock.get('lock_confidence', confidence)
            else:
                # Weak conflicting evidence - ignore it
                if detected_identity != 'Unknown' and detected_identity != locked_identity:
                    print(f"[LOCK] 🔒 Ignoring weak conflict: '{detected_identity}' (conf:{confidence:.3f}) vs locked '{locked_identity}'")
                return locked_identity, track_lock.get('lock_confidence', confidence)
            
            # If still locked, return locked identity
            if track_lock['locked']:
                return locked_identity, track_lock.get('lock_confidence', confidence)
        
        # Not locked - check if we should establish a lock
        if not track_lock['locked'] and detected_identity != 'Unknown' and confidence >= LOCK_CONFIDENCE_THRESHOLD:
            
            # Count recent consistent detections
            recent_detections = list(self.track_detection_history[track_id])[-LOCK_CONSISTENT_FRAMES:]
            
            if len(recent_detections) >= LOCK_CONSISTENT_FRAMES:
                # Check consistency
                same_identity_count = sum(1 for d in recent_detections 
                                        if d['identity'] == detected_identity and d['confidence'] >= LOCK_CONFIDENCE_THRESHOLD)
                
                consistency_ratio = same_identity_count / len(recent_detections)
                
                # Lock if consistent enough - LOWER THE BAR
                if consistency_ratio >= 0.67:  # 67% consistency (2 out of 3 frames)
                    track_lock['locked'] = True
                    track_lock['identity'] = detected_identity
                    track_lock['lock_strength'] = LOCK_CONSISTENT_FRAMES * 2  # Start with stronger lock
                    track_lock['lock_start_frame'] = frame_num
                    track_lock['lock_confidence'] = confidence
                    track_lock['conflicting_detections'] = 0
                    
                    print(f"[LOCK] 🔒 LOCKED onto {detected_identity} for track {track_id} (consistency: {consistency_ratio:.2f})")
                    return detected_identity, confidence
                else:
                    print(f"[DEBUG] Not locking yet - consistency: {consistency_ratio:.2f} (need ≥0.67)")
        
        # Default: return current detection
        track_lock['identity'] = detected_identity if detected_identity != 'Unknown' else track_lock.get('identity', 'Unknown')
        return detected_identity, confidence
    
    def cleanup_old_tracks(self, current_frame, active_track_ids):
        """Clean up locks for tracks that are no longer active"""
        tracks_to_remove = []
        
        for track_id, track_lock in self.track_locks.items():
            # Remove if track is no longer active and hasn't been seen recently
            if track_id not in active_track_ids and \
               current_frame - track_lock['last_seen'] > LOCK_TIMEOUT_FRAMES:
                tracks_to_remove.append(track_id)
        
        for track_id in tracks_to_remove:
            if self.track_locks[track_id]['locked']:
                print(f"[UNLOCK] 🔓 Unlocking {self.track_locks[track_id]['identity']} for track {track_id} (track lost)")
            del self.track_locks[track_id]
            if track_id in self.track_detection_history:
                del self.track_detection_history[track_id]
    
    def find_matching_identity(self, face_encoding, confidence):
        """Find matching identity from global database using face encoding similarity"""
        if face_encoding is None or len(self.global_identities) == 0:
            return None, 0.0
        
        best_match = None
        best_similarity = 0.0
        
        for identity_name, identity_data in self.global_identities.items():
            if 'encodings' not in identity_data or len(identity_data['encodings']) == 0:
                continue
            
            # Compare with stored encodings for this identity
            similarities = []
            for stored_encoding in identity_data['encodings'][-5:]:  # Use last 5 encodings
                # Calculate cosine similarity
                similarity = np.dot(face_encoding, stored_encoding) / (
                    np.linalg.norm(face_encoding) * np.linalg.norm(stored_encoding)
                )
                similarities.append(similarity)
            
            # Use average similarity
            avg_similarity = np.mean(similarities) if similarities else 0.0
            
            if avg_similarity > best_similarity:
                best_similarity = avg_similarity
                best_match = identity_name
        
        # Return match if similarity is above threshold
        if best_similarity > self.identity_persistence_threshold:
            print(f"[DEBUG] Found matching identity: {best_match} (similarity: {best_similarity:.3f})")
            return best_match, best_similarity
        
        return None, 0.0
    
    def update_global_identity(self, name, face_encoding, confidence, frame_num):
        """Update global identity database with new encoding"""
        if name == 'Unknown' or face_encoding is None:
            return
        
        if name not in self.global_identities:
            self.global_identities[name] = {
                'encodings': [],
                'confidence_history': [],
                'last_seen': frame_num
            }
        
        # Add new encoding (keep only recent ones)
        self.global_identities[name]['encodings'].append(face_encoding)
        self.global_identities[name]['confidence_history'].append(confidence)
        self.global_identities[name]['last_seen'] = frame_num
        
        # Limit stored encodings per identity
        if len(self.global_identities[name]['encodings']) > 10:
            self.global_identities[name]['encodings'] = self.global_identities[name]['encodings'][-10:]
            self.global_identities[name]['confidence_history'] = self.global_identities[name]['confidence_history'][-10:]
    
    def process_video_with_tracking(self, video_path=None, use_webcam=False, display=True):
        """Main processing method with identity locking"""
        
        # Initialize video capture
        if use_webcam:
            cap = cv2.VideoCapture(0)
            print("[INFO] Using webcam")
        else:
            cap = cv2.VideoCapture(video_path)
            print(f"[INFO] Using video file: {video_path}")
        
        if not cap.isOpened():
            print("[ERROR] Failed to open video source")
            return
        
        # Processing variables
        track_identities = {}
        track_face_history = defaultdict(lambda: deque(maxlen=300))
        frame_num = 0
        
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                original_frame = frame.copy()
                orig_h, orig_w = original_frame.shape[:2]
                
                # Resize for processing
                if RESIZE_WIDTH and orig_w > RESIZE_WIDTH:
                    ratio = RESIZE_WIDTH / orig_w
                    process_frame = cv2.resize(original_frame, (RESIZE_WIDTH, int(orig_h * ratio)))
                else:
                    process_frame = original_frame
                    ratio = 1.0
                
                # Skip frames for performance
                if frame_num % PROCESS_EVERY_N != 0:
                    frame_num += 1
                    continue
                
                annotated_frame = original_frame.copy()
                
                # YOLO person detection with tracking
                results = self.yolo.track(
                    process_frame,
                    persist=True,
                    classes=[0],  # Only persons
                    conf=PERSON_CONF_THRESHOLD,
                    verbose=False
                )
                
                current_active_tracks = set()
                
                if results and results[0].boxes is not None:
                    boxes = results[0].boxes
                    
                    if boxes.id is not None:
                        track_ids = boxes.id.cpu().numpy().astype(int)
                        current_active_tracks.update(track_ids)
                        bboxes = boxes.xyxy.cpu().numpy()
                        confidences = boxes.conf.cpu().numpy()
                        
                        for track_id, bbox, conf in zip(track_ids, bboxes, confidences):
                            # Convert to original frame coordinates
                            if ratio != 1.0:
                                bbox = bbox / ratio
                            
                            x1, y1, x2, y2 = bbox.astype(int)
                            x1, y1 = max(0, x1), max(0, y1)
                            x2, y2 = min(orig_w, x2), min(orig_h, y2)
                            
                            # Add padding for better face detection
                            padding = 40
                            x1_pad = max(0, x1 - padding)
                            y1_pad = max(0, y1 - padding - 20)
                            x2_pad = min(orig_w, x2 + padding)
                            y2_pad = min(orig_h, y2 + padding + 10)
                            
                            person_crop = original_frame[y1_pad:y2_pad, x1_pad:x2_pad]
                            bbox_for_face = (x1_pad, y1_pad, x2_pad, y2_pad)
                            
                            # Only run face recognition periodically per track
                            should_recognize = (
                                track_id not in self.last_face_recog_frame or 
                                frame_num - self.last_face_recog_frame[track_id] >= FACE_RECOG_EVERY_N
                            )
                            
                            if should_recognize:
                                self.last_face_recog_frame[track_id] = frame_num
                                
                            # Face recognition
                            raw_identity_name = 'Unknown'
                            raw_identity_conf = 0.0
                            face_result = None
                            
                            if should_recognize:
                                if self.debug_count < 2:
                                    print(f"\n=== RECOGNIZING Track {track_id} ===")
                                    self.debug_count += 1
                                
                                face_result = self.face_recognizer.recognize_face_in_crop(
                                    person_crop, original_frame, bbox_for_face
                                )
                                
                                # Get raw recognition result
                                raw_identity_name = face_result.get('name', 'Unknown')
                                raw_identity_conf = face_result.get('confidence', 0.0)
                                face_encoding = face_result.get('face_encoding')
                                
                                # Try global matching if classifier failed
                                if raw_identity_name == 'Unknown' and face_encoding is not None:
                                    matched_name, match_confidence = self.find_matching_identity(face_encoding, raw_identity_conf)
                                    if matched_name:
                                        raw_identity_name = matched_name
                                        raw_identity_conf = match_confidence
                                
                                # Update global database
                                if raw_identity_name != 'Unknown' and raw_identity_conf > RECOG_THRESHOLD:
                                    if face_encoding is not None:
                                        self.update_global_identity(raw_identity_name, face_encoding, raw_identity_conf, frame_num)
                            
                            # Apply locking mechanism
                            final_identity_name, final_identity_conf = self.update_track_lock_status(
                                track_id, raw_identity_name, raw_identity_conf, frame_num
                            )
                            
                            # Basic tracking storage (now uses final identity)
                            if track_id not in track_identities:
                                track_identities[track_id] = {'identity': 'Unknown', 'confidence': 0.0, 'history': []}
                            
                            # Update track with final identity
                            if final_identity_name != 'Unknown':
                                track_identities[track_id]['identity'] = final_identity_name
                                track_identities[track_id]['confidence'] = final_identity_conf
                                track_identities[track_id]['history'].append({
                                    'name': final_identity_name,
                                    'confidence': final_identity_conf,
                                    'frame': frame_num
                                })
                                # Keep only recent history
                                if len(track_identities[track_id]['history']) > 10:
                                    track_identities[track_id]['history'] = track_identities[track_id]['history'][-10:]
                            
                            # Use stored identity if current recognition failed
                            if final_identity_name == 'Unknown' and track_id in track_identities:
                                stored_identity = track_identities[track_id]['identity']
                                if stored_identity != 'Unknown':
                                    final_identity_name = stored_identity
                                    final_identity_conf = track_identities[track_id]['confidence']
                            

                            # Draw results
                            # Color coding: Red=Unknown, Blue=Locked, Yellow=Unlocked but identified
                            is_locked = track_id in self.track_locks and self.track_locks[track_id]['locked']
                            if final_identity_name == 'Unknown':
                                color = (0, 0, 255)  # Red
                            elif is_locked:
                                color = (255, 0, 0)  # Blue for locked
                            else:
                                color = (0, 255, 255)  # Yellow for identified but not locked
                            
                            cv2.rectangle(annotated_frame, (x1, y1), (x2, y2), color, 2)
                            
                            # Draw face box if available
                            if should_recognize and face_result is not None and face_result.get('face_bbox'):
                                fx1, fy1, fx2, fy2 = face_result['face_bbox']
                                cv2.rectangle(annotated_frame, (fx1, fy1), (fx2, fy2), (255, 255, 0), 1)
                            
                            # Enhanced label with lock status
                            lock_indicator = "🔒" if is_locked else ""
                            label = f"ID:{track_id} {lock_indicator}{final_identity_name}"
                            if final_identity_conf > 0:
                                label += f" ({final_identity_conf:.2f})"
                            

                            cv2.putText(annotated_frame, label, (x1, max(30, y1-10)),
                                      cv2.FONT_HERSHEY_SIMPLEX, 0.6, color, 2)
                
                # Clean up old tracks
                self.cleanup_old_tracks(frame_num, current_active_tracks)
                
                # Display
                if display:
                    cv2.imshow("Dlib CNN Recognition", annotated_frame)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                frame_num += 1
                
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()

# -------------------- MAIN --------------------
def main():
    print("[INFO] Starting Dlib CNN recognition system...")
    
    if USE_WEBCAM:
        print("[INFO] Using webcam")
    else:
        if not os.path.exists(VIDEO_PATH):
            print(f"[ERROR] Video file not found: {VIDEO_PATH}")
            return
        print(f"[INFO] Using video file: {VIDEO_PATH}")
    
    system = DlibCNNSystem()
    system.process_video_with_tracking(
        video_path=VIDEO_PATH if not USE_WEBCAM else None,
        use_webcam=USE_WEBCAM,
        display=True
    )

if __name__ == "__main__":
    main()