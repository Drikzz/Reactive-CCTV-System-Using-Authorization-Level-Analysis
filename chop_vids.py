import cv2
import os

def split_video(video_path, label, clip_length=5, output_dir="outputs/clips"):
    # Get the base name of the video (without extension)
    video_name = os.path.splitext(os.path.basename(video_path))[0]

    # Build the output directory structure
    label_dir = os.path.join(output_dir, label, video_name)
    os.makedirs(label_dir, exist_ok=True)

    # Open the video
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        print(f"❌ Cannot open video: {video_path}")
        return

    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    duration = total_frames / fps
    print(f"🎥 Video: {video_name}, Length: {duration:.2f}s, FPS: {fps}, Frames: {total_frames}")

    # Frames per clip
    frames_per_clip = fps * clip_length
    clip_idx = 0
    frame_count = 0

    out = None

    while True:
        ret, frame = cap.read()
        if not ret:
            break

        if frame_count % frames_per_clip == 0:
            # Release previous writer if it exists
            if out:
                out.release()

            clip_idx += 1
            clip_path = os.path.join(label_dir, f"{label}_{clip_idx:04d}.mp4")

            # Define video writer
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            h, w, _ = frame.shape
            out = cv2.VideoWriter(clip_path, fourcc, fps, (w, h))
            print(f"✂️  Creating clip: {clip_path}")

        out.write(frame)
        frame_count += 1

    # Release final writer
    if out:
        out.release()
    cap.release()
    print("✅ Done splitting video!")

if __name__ == "__main__":
    label = input("Enter label for this video: ").strip()
    video_path = input("Enter path to video file: ").strip()
    split_video(video_path, label, clip_length=5)
