"""
Quick Camera Setup and Test Script
Run this to test your camera configuration before using FaceNet
"""

import cv2
import time
import sys
import os

# Add parent directory to path for camera_config import
sys.path.insert(0, os.path.dirname(__file__))

try:
    import camera_config
except ImportError:
    print("[ERROR] Cannot import camera_config.py")
    print("[HELP] Make sure camera_config.py is in the same directory")
    sys.exit(1)


def test_camera():
    """Test the configured camera source"""
    
    camera_config.print_camera_info()
    
    source = camera_config.get_camera_source()
    if source is None:
        print("[ERROR] Failed to get camera source")
        return False
    
    print(f"[INFO] Testing camera: {source}")
    print("[INFO] Opening camera...")
    
    cap = cv2.VideoCapture(source)
    
    # RTSP specific settings
    if camera_config.CAMERA_MODE == "rtsp":
        cap.set(cv2.CAP_PROP_BUFFERSIZE, camera_config.RTSP_BUFFER_SIZE)
        cap.set(cv2.CAP_PROP_OPEN_TIMEOUT_MSEC, camera_config.RTSP_TIMEOUT * 1000)
        cap.set(cv2.CAP_PROP_READ_TIMEOUT_MSEC, camera_config.RTSP_TIMEOUT * 1000)
    
    if not cap.isOpened():
        print("[ERROR] Failed to open camera")
        print("\n[TROUBLESHOOTING]")
        if camera_config.CAMERA_MODE == "rtsp":
            print("1. Check if camera IP is correct")
            print("2. Verify username and password")
            print("3. Ensure camera is on same network")
            print("4. Try pinging the camera IP:")
            camera = camera_config.RTSP_CAMERAS.get(camera_config.ACTIVE_RTSP_CAMERA)
            if camera:
                print(f"   ping {camera['ip']}")
            print(f"5. Test URL in VLC: {source}")
        return False
    
    print("[OK] Camera opened successfully!")
    
    # Get camera properties
    fps = cap.get(cv2.CAP_PROP_FPS)
    width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    print(f"[INFO] Resolution: {width}x{height}")
    print(f"[INFO] FPS: {fps:.1f}")
    
    print("\n[INFO] Capturing frames... Press 'q' to quit")
    
    frame_count = 0
    start_time = time.time()
    last_print_time = start_time
    
    try:
        while True:
            ret, frame = cap.read()
            
            if not ret:
                print("[ERROR] Failed to grab frame")
                if camera_config.CAMERA_MODE == "rtsp":
                    print("[WARN] RTSP connection may be unstable")
                break
            
            frame_count += 1
            
            # Calculate FPS
            elapsed = time.time() - start_time
            current_fps = frame_count / elapsed if elapsed > 0 else 0
            
            # Add info to frame
            cv2.putText(frame, f"FPS: {current_fps:.1f}", (10, 30),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Frame: {frame_count}", (10, 60),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Mode: {camera_config.CAMERA_MODE.upper()}", (10, 90),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            cv2.putText(frame, f"Resolution: {width}x{height}", (10, 120),
                       cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 0), 2)
            
            # Resize for display if needed
            if camera_config.DISPLAY_WIDTH and width > camera_config.DISPLAY_WIDTH:
                scale = camera_config.DISPLAY_WIDTH / width
                new_width = camera_config.DISPLAY_WIDTH
                new_height = int(height * scale)
                frame = cv2.resize(frame, (new_width, new_height))
            
            cv2.imshow("Camera Test - Press 'q' to quit", frame)
            
            # Print stats every 2 seconds
            if time.time() - last_print_time >= 2.0:
                print(f"[INFO] Frames: {frame_count}, FPS: {current_fps:.1f}")
                last_print_time = time.time()
            
            if cv2.waitKey(1) & 0xFF == ord('q'):
                print("\n[INFO] User quit")
                break
                
    except KeyboardInterrupt:
        print("\n[INFO] Interrupted by user")
    except Exception as e:
        print(f"\n[ERROR] Exception during capture: {e}")
    finally:
        cap.release()
        cv2.destroyAllWindows()
    
    print(f"\n[DONE] Captured {frame_count} frames")
    print(f"[DONE] Average FPS: {current_fps:.1f}")
    
    return True


if __name__ == "__main__":
    print("="*60)
    print("CAMERA SETUP AND TEST")
    print("="*60)
    
    success = test_camera()
    
    if success:
        print("\n[SUCCESS] Camera is working!")
        print("[NEXT] You can now run facenet_main.py or facenet_capture.py")
    else:
        print("\n[FAILED] Camera test failed")
        print("[HELP] Check camera_config.py settings and try again")
