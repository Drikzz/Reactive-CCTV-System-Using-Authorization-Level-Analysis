import cv2
import os
import face_recognition
from datetime import datetime

# Ask user for name
name = input("Enter name: ").strip().replace(" ", "_")
user_dir = os.path.join("datasets", "faces", name)
os.makedirs(user_dir, exist_ok=True)

cap = cv2.VideoCapture(0)
cap.set(cv2.CAP_PROP_CONVERT_RGB, 1)  # Ensure RGB color

if not cap.isOpened():
    print("[ERROR] Webcam not found.")
    exit()

print(f"[INFO] Capturing faces for: {name}")
print("[INFO] Press 'q' to quit early.")

count = 0
MAX_IMAGES = 100

while count < MAX_IMAGES:
    ret, frame = cap.read()
    if not ret:
        print("[ERROR] Failed to grab frame.")
        break

    # Convert BGR to RGB for face_recognition
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    # Detect faces
    try:
        face_locations = face_recognition.face_locations(rgb_frame)
    except Exception as e:
        print(f"[ERROR] Face detection failed: {e}")
        continue

    for face_location in face_locations:
        top, right, bottom, left = face_location
        face_image = frame[top:bottom, left:right]

        if face_image.size == 0:
            continue

        filename = f"{name}_{count:03d}.jpg"
        path = os.path.join(user_dir, filename)
        cv2.imwrite(path, face_image)
        print(f"[INFO] Saved {filename}")
        count += 1

        if count >= MAX_IMAGES:
            break

    # Show frame with rectangle
    for (top, right, bottom, left) in face_locations:
        cv2.rectangle(frame, (left, top), (right, bottom), (0, 255, 0), 2)

    cv2.imshow("Face Capture", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print(f"[DONE] Collected {count} face images in: {user_dir}")
