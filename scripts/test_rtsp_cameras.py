"""Quick test to verify RTSP camera connections"""
import cv2
import sys
from pathlib import Path

# Add parent directory to path
REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT / "scripts"))

import camera_config_streamlit as cam_config

def test_rtsp_camera(camera_key):
    """Test a single RTSP camera"""
    print(f"\n{'='*60}")
    print(f"Testing: {camera_key}")
    print(f"{'='*60}")
    
    url = cam_config.get_rtsp_url(camera_key)
    if url is None:
        print(f"❌ Failed to generate RTSP URL for {camera_key}")
        return False
    
    print(f"URL: {url}")
    print("Attempting to connect...")
    
    cap = cv2.VideoCapture(url)
    cap.set(cv2.CAP_PROP_BUFFERSIZE, 1)
    
    if not cap.isOpened():
        print(f"❌ Failed to open camera: {camera_key}")
        cap.release()
        return False
    
    print("✅ Camera opened successfully!")
    print("Reading test frame...")
    
    ret, frame = cap.read()
    if not ret or frame is None:
        print(f"❌ Failed to read frame from {camera_key}")
        cap.release()
        return False
    
    h, w = frame.shape[:2]
    print(f"✅ Successfully read frame: {w}x{h}")
    print(f"✅ Camera {camera_key} is working!")
    
    cap.release()
    return True

def main():
    print("\n" + "="*60)
    print("RTSP CAMERA CONNECTION TEST")
    print("="*60)
    
    cameras = cam_config.get_all_rtsp_cameras()
    
    if not cameras:
        print("❌ No RTSP cameras configured")
        return
    
    print(f"\nFound {len(cameras)} camera(s) in configuration:\n")
    
    results = {}
    for key, name, enabled in cameras:
        status = "✓ Enabled" if enabled else "✗ Disabled"
        print(f"  {status}: {key} - {name}")
        
        if enabled:
            results[key] = test_rtsp_camera(key)
    
    # Summary
    print("\n" + "="*60)
    print("SUMMARY")
    print("="*60)
    
    for key, success in results.items():
        status = "✅ WORKING" if success else "❌ FAILED"
        print(f"{status}: {key}")
    
    if all(results.values()):
        print("\n🎉 All cameras are working!")
    else:
        print("\n⚠️ Some cameras failed to connect.")
        print("\nTroubleshooting tips:")
        print("1. Check camera IP addresses are correct")
        print("2. Verify username/password in camera_config_streamlit.py")
        print("3. Ensure cameras are on the same network")
        print("4. Try accessing RTSP URL in VLC: Media > Open Network Stream")

if __name__ == "__main__":
    main()
