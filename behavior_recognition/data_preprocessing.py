import os
import cv2
import numpy as np

def load_ucf101_dataset(dataset_dir='datasets/ucf101', num_frames=16, img_size=(224, 224)):
    videos, labels = [], []
    categories = sorted(os.listdir(dataset_dir))

    for label, category in enumerate(categories):
        category_path = os.path.join(dataset_dir, category)
        if not os.path.isdir(category_path):
            continue
        for video_file in os.listdir(category_path):
            if not video_file.endswith('.avi'):
                continue
            video_path = os.path.join(category_path, video_file)
            frames = extract_frames(video_path, num_frames, img_size)
            if frames is not None:
                videos.append(frames)
                labels.append(label)

    return np.array(videos), np.array(labels)

def extract_frames(video_path, num_frames=16, img_size=(224, 224)):
    cap = cv2.VideoCapture(video_path)
    total = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    interval = max(total // num_frames, 1)
    frames = []

    for i in range(num_frames):
        cap.set(cv2.CAP_PROP_POS_FRAMES, i * interval)
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.resize(frame, img_size)
        frames.append(frame)
    cap.release()

    if len(frames) == num_frames:
        return np.array(frames)
    return None