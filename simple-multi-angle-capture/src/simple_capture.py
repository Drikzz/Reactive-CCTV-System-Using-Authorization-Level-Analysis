"""
Simple Multi-Angle Capture System
This script captures images from a webcam at multiple angles for better recognition training.
"""

import os
import cv2
import time
import sys

# Configuration
ANGLE_SEQUENCE = [
    "frontal", 
    "left_profile", 
    "right_profile", 
    "quarter_left", 
    "quarter_right", 
    "up", 
    "down"
]
CAPTURES_PER_ANGLE = 20

def ensure_dirs(base_dir, name):
    """Create directory structure for all angles."""
    folders = {}
    for angle in ANGLE_SEQUENCE:
        folder_path = os.path.join(base_dir, name, angle)
        os.makedirs(folder_path, exist_ok=True)
        folders[angle] = folder_path
    return folders

def is_good_quality(img, min_brightness=50, blur_threshold=100):
    """Check if image has good quality for face recognition."""
    if img is None or img.size == 0:
        return False
    
    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    # Check brightness
    mean_brightness = cv2.mean(gray)[0]
    if mean_brightness < min_brightness:
        return False
    
    # Check blur using Laplacian variance on grayscale
    laplacian_var = cv2.Laplacian(gray, cv2.CV_64F).var()
    if laplacian_var < blur_threshold:
        return False
    
    return True

def capture_multi_angle_faces(source=0, name=None, out_dir="datasets/faces", require_quality=True, suppress_messages=False):
    """Capture faces at multiple angles.

    New params:
      - require_quality (bool): if False, save frames regardless of quality checks
      - suppress_messages (bool): if True, don't print quality-related messages
    Controls (while preview window is active):
      - SPACE: capture one frame
      - a     : start automatic capture for this angle (CAPTURES_PER_ANGLE)
      - n     : skip to next angle
      - q     : quit entirely
    """
    
    if name is None:
        name = input("Enter your name: ").strip().replace(" ", "_")
    
    folders = ensure_dirs(out_dir, name)
    if not suppress_messages:
        print(f"Saving to: {os.path.join(out_dir, name)}")
    
    cap = cv2.VideoCapture(source)
    if not cap.isOpened():
        if not suppress_messages:
            print("Unable to open video source.")
        return

    window_name = "Multi-Angle Capture - SPACE capture | a auto | n next | q quit"
    cv2.namedWindow(window_name, cv2.WINDOW_NORMAL)

    try:
        for angle in ANGLE_SEQUENCE:
            if not suppress_messages:
                print(f"\nPosition your head for the {angle} angle.")
                print("Controls: SPACE capture | a auto-capture | n next angle | q quit")
            saved_count = 0

            auto_mode = False
            last_auto_time = 0.0
            auto_interval = 0.25  # seconds between auto captures

            while True:
                ret, frame = cap.read()
                if not ret:
                    if not suppress_messages:
                        print("Cannot read frame from camera.")
                    break

                # overlay instructions and status
                overlay = frame.copy()
                h, w = overlay.shape[:2]
                status = f"Angle: {angle}  Saved: {saved_count}/{CAPTURES_PER_ANGLE}"
                cv2.putText(overlay, status, (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.8, (0,255,0), 2)
                cv2.putText(overlay, "SPACE capture | a auto | n next | q quit", (10, h-10),
                            cv2.FONT_HERSHEY_SIMPLEX, 0.6, (255,255,255), 1)
                
                cv2.imshow(window_name, overlay)
                key = cv2.waitKey(1) & 0xFF

                if auto_mode:
                    now = time.time()
                    if now - last_auto_time >= auto_interval:
                        last_auto_time = now
                        if require_quality:
                            if is_good_quality(frame):
                                filename = os.path.join(folders[angle], f"{name}_{angle}_{saved_count+1}.jpg")
                                cv2.imwrite(filename, frame)
                                saved_count += 1
                                if not suppress_messages:
                                    print(f"Auto captured: {filename}")
                            else:
                                if not suppress_messages:
                                    print("Auto-skip: poor quality")
                        else:
                            filename = os.path.join(folders[angle], f"{name}_{angle}_{saved_count+1}.jpg")
                            cv2.imwrite(filename, frame)
                            saved_count += 1
                            if not suppress_messages:
                                print(f"Auto captured (forced): {filename}")
                    if saved_count >= CAPTURES_PER_ANGLE:
                        auto_mode = False
                        if not suppress_messages:
                            print(f"Auto capture done for angle '{angle}'")
                        break

                if key == ord(' '):  # manual capture
                    if require_quality:
                        if is_good_quality(frame):
                            filename = os.path.join(folders[angle], f"{name}_{angle}_{saved_count+1}.jpg")
                            cv2.imwrite(filename, frame)
                            saved_count += 1
                            if not suppress_messages:
                                print(f"Captured: {filename}")
                        else:
                            if not suppress_messages:
                                print("Image quality is not good enough, skipped.")
                    else:
                        filename = os.path.join(folders[angle], f"{name}_{angle}_{saved_count+1}.jpg")
                        cv2.imwrite(filename, frame)
                        saved_count += 1
                        if not suppress_messages:
                            print(f"Captured (forced): {filename}")

                    if saved_count >= CAPTURES_PER_ANGLE:
                        if not suppress_messages:
                            print(f"Reached {CAPTURES_PER_ANGLE} captures for angle '{angle}'")
                        break

                elif key == ord('a'):  # start auto capture
                    if not suppress_messages:
                        print("Starting automatic capture...")
                    auto_mode = True
                    last_auto_time = time.time()

                elif key == ord('n'):  # next angle
                    if not suppress_messages:
                        print(f"Skipping to next angle from '{angle}'")
                    break

                elif key == ord('q'):  # quit entire process
                    if not suppress_messages:
                        print("Quitting capture.")
                    cap.release()
                    cv2.destroyAllWindows()
                    return

            # small pause between angles
            time.sleep(0.5)

    finally:
        cap.release()
        cv2.destroyAllWindows()

    if not suppress_messages:
        print("Capture complete.")

src = os.path.dirname(__file__)
if src not in sys.path:
    sys.path.insert(0, src)
from simple_capture import capture_multi_angle_faces

if __name__ == "__main__":
    capture_multi_angle_faces()