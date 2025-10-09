import cv2
import pyvirtualcam

# --- CAMERA CONNECTION SETUP ---
ip_address = '192.168.1.244'  # replace with your Tapo camera IP
port = '554'
username = 'reactivecctv101'
password = 'reactivecctv101'

# 1080p stream1 or 480p stream2
rtsp_url = f"rtsp://{username}:{password}@{ip_address}:{port}/stream2"

# Open RTSP Stream
cap = cv2.VideoCapture(rtsp_url)
if not cap.isOpened():
    print("❌ Failed to open RTSP stream")
    exit()

# Get stream dimensions
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH)) or 640
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT)) or 480
fps = int(cap.get(cv2.CAP_PROP_FPS)) or 30

print(f"Connected to RTSP: {width}x{height} @ {fps} FPS")

# --- VIRTUAL CAMERA SETUP ---
with pyvirtualcam.Camera(width=width, height=height, fps=fps, fmt=pyvirtualcam.PixelFormat.BGR) as cam:
    print(f"🎥 Virtual camera active: {cam.device}")
    print("Press 'q' to quit")

    while True:
        ret, frame = cap.read()
        if not ret:
            print("⚠️ Failed to read frame")
            break

        # Optional: add processing (flip, filter, overlay, etc.)
        # frame = cv2.flip(frame, 1)

        # Send frame to virtual cam
        cam.send(frame)
        cam.sleep_until_next_frame()

        # Also show a preview
        cv2.imshow("Tapo RTSP Feed", frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

cap.release()
cv2.destroyAllWindows()
