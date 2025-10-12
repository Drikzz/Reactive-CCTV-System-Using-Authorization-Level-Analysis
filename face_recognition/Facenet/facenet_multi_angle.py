"""
FaceNet Multi-Angle Face Capture System
Based on facenet_capture.py but captures multiple angles for better recognition and training
"""

import os
import sys
import cv2
import numpy as np
import time
from datetime import datetime
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization
import torch
import argparse
from sklearn.metrics.pairwise import cosine_similarity

# Add the root project dir to sys.path
project_root = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


sys.path.append(project_root)

# Configuration
FACE_SIZE = (160, 160)
CAPTURES_PER_ANGLE = 20
MIN_FACE_SIZE = 40
CAPTURE_COOLDOWN = 0.2

# Define angle sequence for systematic capture
ANGLE_SEQUENCE = [
    "frontal", 
    "left_profile", 
    "right_profile", 
    "quarter_left", 
    "quarter_right", 
    "up", 
    "down"
]

# Angle targets with descriptions
ANGLE_TARGETS = {
    "frontal": {
        "description": "Face the camera directly",
        "color": (0, 255, 0)  # Green
    },
    "left_profile": {
        "description": "Turn head left (your left)",
        "color": (255, 0, 0)  # Blue
    },
    "right_profile": {
        "description": "Turn head right (your right)", 
        "color": (0, 0, 255)  # Red
    },
    "quarter_left": {
        "description": "Turn head slightly left",
        "color": (255, 255, 0)  # Cyan
    },
    "quarter_right": {
        "description": "Turn head slightly right",
        "color": (0, 255, 255)  # Yellow
    },
    "up": {
        "description": "Tilt head up slightly",
        "color": (255, 0, 255)  # Magenta
    },
    "down": {
        "description": "Tilt head down slightly",
        "color": (128, 255, 128)  # Light green
    }
}

def ensure_dirs(base_dir, name):
    """Create directory structure for all angles"""
    folders = {}
    for angle in ANGLE_SEQUENCE:
        folder_path = os.path.join(base_dir, name, angle)
        os.makedirs(folder_path, exist_ok=True)
        folders[angle] = folder_path
    return folders

def is_good_quality(img, min_brightness=50, max_brightness=200, blur_threshold=100):
    """Check if image has good quality for face recognition"""
    if img is None or img.size == 0:
        return False, "empty"
    
    # Convert to grayscale for quality checks
    if len(img.shape) == 3:
        gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    else:
        gray = img
    
    # Check brightness
    mean_brightness = np.mean(gray)
    if mean_brightness < min_brightness:
        return False, "too dark"
    if mean_brightness > max_brightness:
        return False, "too bright"
    
    # Check blur using Laplacian variance
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False, "too blurry"
    
    return True, ""

def align_face(img, landmarks, output_size=160):
    """Simple face alignment using landmarks"""
    try:
        # Get eye positions (landmarks 36-41 for left eye, 42-47 for right eye)
        left_eye = np.mean(landmarks[36:42], axis=0)
        right_eye = np.mean(landmarks[42:48], axis=0)
        
        # Calculate angle between eyes
        eye_center = ((left_eye[0] + right_eye[0]) * 0.5, (left_eye[1] + right_eye[1]) * 0.5)
        dy = right_eye[1] - left_eye[1]
        dx = right_eye[0] - left_eye[0]
        angle = np.degrees(np.arctan2(dy, dx))
        
        # Get rotation matrix
        M = cv2.getRotationMatrix2D(eye_center, angle, 1)
        
        # Rotate image
        rotated = cv2.warpAffine(img, M, (img.shape[1], img.shape[0]))
        
        # Crop and resize
        h, w = rotated.shape[:2]
        center_x, center_y = int(eye_center[0]), int(eye_center[1])
        
        # Calculate crop region
        crop_size = min(h, w) // 2
        x1 = max(0, center_x - crop_size//2)
        y1 = max(0, center_y - crop_size//2)
        x2 = min(w, x1 + crop_size)
        y2 = min(h, y1 + crop_size)
        
        cropped = rotated[y1:y2, x1:x2]
        aligned = cv2.resize(cropped, (output_size, output_size))
        
        return aligned
    except:
        return None

def capture_multi_angle_faces(source=0, name=None, out_dir="datasets/faces", device=None):
    """Capture faces at multiple angles for better training"""
    
    device = device or ("cuda" if torch.cuda.is_available() else "cpu")
    mtcnn = MTCNN(image_size=FACE_SIZE[0], margin=0, keep_all=True, device=device, post_process=False, min_face_size=MIN_FACE_SIZE)
    
    # Ask for name interactively
    if name is None:
        print("\n" + "="*50)
        print("🎥 MULTI-ANGLE FACE CAPTURE")
        print("="*50)
        name = input("👤 Enter person name: ").strip().replace(" ", "_")
        if not name:
            print("[ERROR] Name cannot be empty!")
            return
    
    folders = ensure_dirs(out_dir, name)
    print(f"\n[INFO] 🎯 Device: {device}")
    print(f"[INFO] 📁 Saving to: {os.path.join(out_dir, name)}")
    print(f"[INFO] 📸 Will capture {CAPTURES_PER_ANGLE} images per angle when you press ENTER")
    print("[INFO] ⌨️  Controls:")
    print("       ENTER - Capture 20 images for current angle")
    print("       'n' - Next angle")
    print("       'p' - Previous angle")
    print("       'q' - Quit")
    print("       'r' - Reset counters")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        print("[ERROR] Unable to open video source:", source)
        return

    # Tracking variables
    counters = {angle: len(os.listdir(folders[angle])) for angle in ANGLE_SEQUENCE}
    total_saved = 0
    capturing = False
    current_angle_index = 0
    
    print(f"\n[INFO] 📊 Starting counts: {counters}")
    
    # Instructions
    print("\n[INFO] 🎬 Instructions:")
    print("  1. Position your head at the desired angle")
    print("  2. Use 'n'/'p' to navigate to the angle you want to capture")
    print("  3. Press ENTER to capture 20 images")
    print("  4. Move to next angle and repeat")
    print("\n[INFO] Angle sequence:")
    for i, angle in enumerate(ANGLE_SEQUENCE):
        remaining = CAPTURES_PER_ANGLE - counters[angle]
        status = "✅ Complete" if remaining <= 0 else f"📸 Need {remaining}"
        current_marker = "👉 " if i == current_angle_index else "   "
        description = ANGLE_TARGETS[angle]["description"]
        print(f"  {current_marker}{angle}: {description} - {status}")
    
    print(f"\n🚀 Ready! Current angle: {ANGLE_SEQUENCE[current_angle_index]}")
    print("Press ENTER to capture 20 images...")
    
    while True:
        ret, frame = cap.read()
        if not ret:
            print("[ERROR] Cannot read frame")
            break
        
        # Display frame
        display_frame = frame.copy()
        
        # Current angle being captured
        current_angle = ANGLE_SEQUENCE[current_angle_index]
        
        # Draw status info
        capture_status = "CAPTURING..." if capturing else f"Ready for {current_angle} - Press ENTER"
        cv2.putText(display_frame, f"Multi-Angle Capture: {name}", (10, 30), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        
        cv2.putText(display_frame, capture_status, (10, 60), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 255) if not capturing else (0, 255, 0), 2)
        
        cv2.putText(display_frame, f"Total saved: {total_saved}", (10, 90), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show current angle info
        angle_desc = ANGLE_TARGETS[current_angle]["description"]
        cv2.putText(display_frame, f"Current: {current_angle} - {angle_desc}", (10, 120), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
        cv2.putText(display_frame, "Use 'n'/'p' to change angle, ENTER to capture", (10, 140), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.4, (255, 255, 255), 1)
        
        # Convert to RGB for MTCNN
        rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        
        try:
            boxes, probs, landmarks_all = mtcnn.detect(rgb, landmarks=True)
        except Exception as e:
            boxes, probs, landmarks_all = None, None, None
        
        # Process faces
        if boxes is not None and len(boxes) > 0:
            for i, box in enumerate(boxes):
                prob = float(probs[i]) if probs is not None else 0.0
                if prob < 0.5:
                    continue
                
                x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                
                # Draw face box with current angle color
                box_color = ANGLE_TARGETS[current_angle]["color"]
                cv2.rectangle(display_frame, (x1, y1), (x2, y2), box_color, 3)
                
                # Show capture info
                remaining = CAPTURES_PER_ANGLE - counters[current_angle]
                if remaining > 0:
                    cv2.putText(display_frame, f"READY: Press ENTER", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
                    cv2.putText(display_frame, f"Will capture {remaining} for {current_angle}", (x1, y2+45), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.5, (255, 255, 0), 1)
                else:
                    cv2.putText(display_frame, f"{current_angle} COMPLETE", (x1, y2+20), 
                               cv2.FONT_HERSHEY_SIMPLEX, 0.6, (0, 255, 0), 2)
        
        # Show angle progress on side
        y_offset = 160
        cv2.putText(display_frame, "Angle Progress:", (10, y_offset), 
                   cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255, 255, 255), 2)
        
        for i, angle in enumerate(ANGLE_SEQUENCE):
            y_offset += 25
            count = counters[angle]
            status = "✅" if count >= CAPTURES_PER_ANGLE else f"{count}/{CAPTURES_PER_ANGLE}"
            color = (0, 255, 0) if count >= CAPTURES_PER_ANGLE else ANGLE_TARGETS[angle]["color"]
            
            # Highlight current angle
            text = f"{angle}: {status}"
            if i == current_angle_index:
                text = f">>> {text} <<<"
                cv2.putText(display_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 2)
            else:
                cv2.putText(display_frame, text, (10, y_offset), 
                           cv2.FONT_HERSHEY_SIMPLEX, 0.5, color, 1)
        
        cv2.imshow("Multi-Angle Face Capture", display_frame)
        
        # Check for completion
        if all(counters[angle] >= CAPTURES_PER_ANGLE for angle in ANGLE_SEQUENCE):
            print(f"\n🎉 [COMPLETED] All angles captured! Total: {total_saved} images")
            break
        
        # Handle keyboard input
        key = cv2.waitKey(1) & 0xFF
        
        if key == 13:  # Enter key
            if not capturing:
                remaining = CAPTURES_PER_ANGLE - counters[current_angle]
                if remaining > 0:
                    print(f"\n🚀 [CAPTURING] Starting capture for {current_angle}...")
                    print(f"    Will capture {remaining} images rapidly...")
                    capturing = True
                    capture_count = 0
                    capture_start_time = time.time()
                    
                    # Rapid capture loop
                    while capture_count < remaining and capturing:
                        ret, capture_frame = cap.read()
                        if not ret:
                            break
                        
                        # Convert to RGB for MTCNN
                        rgb = cv2.cvtColor(capture_frame, cv2.COLOR_BGR2RGB)
                        
                        try:
                            boxes, probs, landmarks_all = mtcnn.detect(rgb, landmarks=True)
                        except:
                            continue
                        
                        if boxes is not None and len(boxes) > 0:
                            box = boxes[0]  # Use first face
                            prob = float(probs[0]) if probs is not None else 0.0
                            
                            if prob > 0.5:
                                x1, y1, x2, y2 = [int(max(0, v)) for v in box]
                                
                                # Extract and process face
                                pad = int(0.15 * max(1, (x2 - x1)))
                                xx1, yy1 = max(0, x1 - pad), max(0, y1 - pad)
                                xx2, yy2 = min(capture_frame.shape[1], x2 + pad), min(capture_frame.shape[0], y2 + pad)
                                face_crop = capture_frame[yy1:yy2, xx1:xx2].copy()
                                
                                # Try alignment
                                aligned = None
                                lm = landmarks_all[0] if landmarks_all is not None else None
                                if lm is not None:
                                    try:
                                        aligned = align_face(capture_frame, lm, output_size=FACE_SIZE[0])
                                    except:
                                        aligned = None
                                
                                if aligned is None and face_crop.size > 0:
                                    aligned = cv2.resize(face_crop, FACE_SIZE)
                                
                                if aligned is not None:
                                    good, reason = is_good_quality(aligned)
                                    
                                    if good:
                                        # Save the image
                                        ts = datetime.now().strftime("%Y%m%d_%H%M%S_%f")
                                        current_count = counters[current_angle] + capture_count + 1
                                        fname = f"{name}_{current_angle}_{current_count:02d}_{ts}.jpg"
                                        out_path = os.path.join(folders[current_angle], fname)
                                        
                                        try:
                                            cv2.imwrite(out_path, aligned)
                                            capture_count += 1
                                            total_saved += 1
                                            print(f"    📸 [{capture_count}/{remaining}] {fname}")
                                            
                                            # Brief pause between captures
                                            time.sleep(CAPTURE_COOLDOWN)
                                            
                                        except Exception as e:
                                            print(f"    ❌ Failed to save: {e}")
                        
                        # Check for early termination
                        key_check = cv2.waitKey(1) & 0xFF
                        if key_check == ord('q'):
                            capturing = False
                            break
                    
                    # Update counter and finish
                    counters[current_angle] += capture_count
                    capture_time = time.time() - capture_start_time
                    capturing = False
                    
                    print(f"✅ [FINISHED] Captured {capture_count} images for {current_angle} in {capture_time:.1f}s")
                    
                    if counters[current_angle] >= CAPTURES_PER_ANGLE:
                        print(f"🎯 [COMPLETE] {current_angle} angle finished!")
                        
                        # Auto-advance to next incomplete angle
                        next_angle_found = False
                        for i in range(len(ANGLE_SEQUENCE)):
                            next_index = (current_angle_index + 1 + i) % len(ANGLE_SEQUENCE)
                            if counters[ANGLE_SEQUENCE[next_index]] < CAPTURES_PER_ANGLE:
                                current_angle_index = next_index
                                print(f"🔄 Moving to next angle: {ANGLE_SEQUENCE[current_angle_index]}")
                                next_angle_found = True
                                break
                        
                        if not next_angle_found:
                            print("🎉 All angles complete!")
                else:
                    print(f"ℹ️  {current_angle} already complete ({counters[current_angle]}/{CAPTURES_PER_ANGLE})")
            else:
                print("⚠️  Currently capturing, please wait...")
        
        elif key == ord('q'):
            print("\n[INFO] Quitting...")
            break
        elif key == ord('r'):
            # Reset counters
            for angle in ANGLE_SEQUENCE:
                counters[angle] = 0
            total_saved = 0
            current_angle_index = 0
            print("[INFO] Counters reset")
        elif key == ord('n'):
            # Manual next angle
            if not capturing:
                current_angle_index = (current_angle_index + 1) % len(ANGLE_SEQUENCE)
                current_angle = ANGLE_SEQUENCE[current_angle_index]
                angle_desc = ANGLE_TARGETS[current_angle]["description"]
                print(f"[INFO] Switched to: {current_angle} - {angle_desc}")
        elif key == ord('p'):
            # Manual previous angle
            if not capturing:
                current_angle_index = (current_angle_index - 1) % len(ANGLE_SEQUENCE)
                current_angle = ANGLE_SEQUENCE[current_angle_index]
                angle_desc = ANGLE_TARGETS[current_angle]["description"]
                print(f"[INFO] Switched to: {current_angle} - {angle_desc}")
    
    cap.release()
    cv2.destroyAllWindows()
    
    print(f"\n📊 [FINAL RESULTS]")
    for angle in ANGLE_SEQUENCE:
        count = len(os.listdir(folders[angle]))
        status = "✅ Complete" if count >= CAPTURES_PER_ANGLE else f"⚠️  Incomplete"
        print(f"  {angle}: {count} images {status}")
    
    print(f"\n📁 All images saved to: {os.path.join(out_dir, name)}")
    print("🚀 Ready for training with multi-angle face data!")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Multi-angle face capture for better recognition training")
    parser.add_argument("--source", "-s", default=0, help="video source (0 for webcam or path)")
    parser.add_argument("--out", "-o", default=os.path.join("datasets","faces"), help="output base directory")
    parser.add_argument("--device", "-d", default=None, help="torch device: cpu or cuda")
    args = parser.parse_args()

    print("=== MULTI-ANGLE FACE CAPTURE ===")
    print("Capture faces at different angles for better recognition and training!")
    print("Based on facenet_capture.py with multi-angle support.")
    print(f"\nAngles to capture: {', '.join(ANGLE_SEQUENCE)}")
    print(f"Images per angle: {CAPTURES_PER_ANGLE}")
    print("\nControls:")
    print("  ENTER - Capture 20 images for current angle")
    print("  'n' - Next angle")
    print("  'p' - Previous angle") 
    print("  'r' - Reset counters")
    print("  'q' - Quit")
    
    try:
        capture_multi_angle_faces(source=args.source, name=None, out_dir=args.out, device=args.device)
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"[ERROR] {e}")