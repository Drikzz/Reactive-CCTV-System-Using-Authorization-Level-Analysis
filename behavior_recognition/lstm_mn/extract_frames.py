"""
Step 1: Extract and Preprocess Frames from Videos with Person Tracking
Uses YOLOv8 to track individual persons and crop them from each frame.
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
from collections import defaultdict
import config


def extract_person_crops_from_video(video_path, model, output_dir, 
                                     enable_tracking=True, max_persons=2):
    """
    Extract frames from video and crop detected persons using YOLO tracking.
    
    Args:
        video_path: Path to input video
        model: YOLOv8 detection model
        output_dir: Directory to save cropped frames
        enable_tracking: Whether to use YOLO tracking (vs simple detection)
        max_persons: Maximum number of persons to track
        
    Returns:
        Dictionary mapping person_id -> list of frame paths
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"❌ Error: Cannot open video {video_path}")
        return {}
    
    # Get video properties
    fps = int(cap.get(cv2.CAP_PROP_FPS))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    frame_skip = max(1, fps // config.TARGET_FPS)  # Skip frames to achieve target FPS
    
    print(f"   Video FPS: {fps}, Target FPS: {config.TARGET_FPS}, Frame skip: {frame_skip}")
    
    # Track persons and their frame sequences
    person_frames = defaultdict(list)
    frame_count = 0
    saved_frame_count = 0
    
    with tqdm(total=total_frames, desc=f"   Processing {video_path.name}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Skip frames to match target FPS
            if frame_count % frame_skip != 0:
                frame_count += 1
                pbar.update(1)
                continue
            
            # Run YOLOv8 detection/tracking
            if enable_tracking:
                # Use .track() for consistent person IDs across frames
                results = model.track(
                    frame, 
                    conf=config.YOLO_CONF_THRESHOLD,
                    iou=config.YOLO_IOU_THRESHOLD,
                    persist=True,  # Persist tracks between frames
                    classes=[0],  # Only detect persons (class 0)
                    verbose=False
                )
            else:
                # Simple detection without tracking
                results = model(
                    frame,
                    conf=config.YOLO_CONF_THRESHOLD,
                    classes=[0],
                    verbose=False
                )
            
            # Process detections
            if results[0].boxes is not None and len(results[0].boxes) > 0:
                boxes = results[0].boxes.xyxy.cpu().numpy()  # Bounding boxes
                confidences = results[0].boxes.conf.cpu().numpy()  # Confidence scores
                
                # Get track IDs if tracking is enabled
                if enable_tracking and hasattr(results[0].boxes, 'id') and results[0].boxes.id is not None:
                    track_ids = results[0].boxes.id.cpu().numpy().astype(int)
                else:
                    # Assign simple IDs based on detection order
                    track_ids = np.arange(len(boxes))
                
                # Sort by confidence and take top N persons
                sorted_indices = np.argsort(confidences)[::-1][:max_persons]
                
                for idx in sorted_indices:
                    if confidences[idx] < config.MIN_DETECTION_CONFIDENCE:
                        continue
                    
                    person_id = track_ids[idx]
                    x1, y1, x2, y2 = boxes[idx].astype(int)
                    
                    # Ensure bounding box is within frame
                    x1, y1 = max(0, x1), max(0, y1)
                    x2, y2 = min(frame.shape[1], x2), min(frame.shape[0], y2)
                    
                    # Crop person from frame
                    person_crop = frame[y1:y2, x1:x2]
                    
                    if person_crop.size == 0:
                        continue
                    
                    # Resize to MobileNetV2 input size
                    person_crop_resized = cv2.resize(
                        person_crop, 
                        (config.FRAME_WIDTH, config.FRAME_HEIGHT),
                        interpolation=cv2.INTER_LINEAR
                    )
                    
                    # Create person directory
                    person_dir = output_dir / f"person_{person_id}"
                    person_dir.mkdir(parents=True, exist_ok=True)
                    
                    # Save frame with zero-padded numbering
                    frame_filename = f"frame{saved_frame_count:04d}.png"
                    frame_path = person_dir / frame_filename
                    cv2.imwrite(str(frame_path), person_crop_resized)
                    
                    person_frames[person_id].append(frame_path)
            
            saved_frame_count += 1
            frame_count += 1
            pbar.update(1)
    
    cap.release()
    
    # Filter out tracks that are too short
    valid_person_frames = {
        pid: frames for pid, frames in person_frames.items()
        if len(frames) >= config.MIN_TRACK_LENGTH
    }
    
    return valid_person_frames


def process_all_videos():
    """
    Process all videos in behavior_clips directory and extract person-tracked frames.
    """
    print("=" * 60)
    print("FRAME EXTRACTION WITH PERSON TRACKING PIPELINE")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load YOLOv8 detection model
    print(f"\n📦 Loading YOLOv8 detection model from {config.YOLO_DETECTION_MODEL}")
    model = YOLO(str(config.YOLO_DETECTION_MODEL))
    model.to(device)
    print("✓ Model loaded successfully")
    
    # Display extraction settings
    print(f"\n⚙️  Extraction Settings:")
    print(f"   Target FPS: {config.TARGET_FPS}")
    print(f"   Frame size: {config.FRAME_WIDTH}x{config.FRAME_HEIGHT}")
    print(f"   Sequence length: {config.SEQUENCE_LENGTH} frames")
    print(f"   Person tracking: {'Enabled' if config.ENABLE_PERSON_TRACKING else 'Disabled'}")
    print(f"   Max persons per video: {config.MAX_PERSONS_PER_VIDEO}")
    print(f"   Min track length: {config.MIN_TRACK_LENGTH} frames")
    print(f"   YOLO confidence: {config.YOLO_CONF_THRESHOLD}")
    
    # Check if input directory exists
    if not config.BEHAVIOR_CLIPS_DIR.exists():
        print(f"\n❌ Error: Input directory not found: {config.BEHAVIOR_CLIPS_DIR}")
        print("Please create the directory and add video files organized by class.")
        return
    
    # Get all class directories from multiple datasets
    class_dirs = []
    for dataset_name in config.BEHAVIOR_DATASETS:
        dataset_path = config.BEHAVIOR_CLIPS_DIR / dataset_name
        if dataset_path.exists():
            dataset_class_dirs = [d for d in dataset_path.iterdir() if d.is_dir()]
            class_dirs.extend([(d, dataset_name) for d in dataset_class_dirs])
            print(f"   Found {len(dataset_class_dirs)} classes in {dataset_name}")
        else:
            print(f"⚠️  Warning: Dataset directory not found: {dataset_path}")
    
    if len(class_dirs) == 0:
        print(f"\n❌ No class directories found in any dataset")
        print("Expected structure: behavior_clips/suspicious-actions/<class_name>/*.mp4")
        print("                   behavior_clips/neutral-actions/<class_name>/*.mp4")
        return
    
    print(f"\n📁 Found {len(class_dirs)} behavior classes")
    
    total_processed = 0
    total_failed = 0
    total_persons = 0
    class_summary = {}
    
    # Process each class
    for class_dir, dataset_name in class_dirs:
        class_name = class_dir.name
        print(f"\n{'=' * 60}")
        print(f"Processing class: {class_name} (from {dataset_name})")
        print(f"{'=' * 60}")
        
        # Get all video files
        video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        
        if len(video_files) == 0:
            print(f"⚠️  No video files found in {class_dir}")
            continue
        
        print(f"Found {len(video_files)} videos")
        
        processed = 0
        failed = 0
        persons_found = 0
        
        # Process each video
        for video_path in video_files:
            try:
                # Create output directory for this video
                output_dir = config.BEHAVIOR_FRAMES_DIR / dataset_name / class_name / video_path.stem
                output_dir.mkdir(parents=True, exist_ok=True)
                
                # Extract person crops
                person_frames = extract_person_crops_from_video(
                    video_path,
                    model,
                    output_dir,
                    enable_tracking=config.ENABLE_PERSON_TRACKING,
                    max_persons=config.MAX_PERSONS_PER_VIDEO
                )
                
                if len(person_frames) > 0:
                    processed += 1
                    persons_found += len(person_frames)
                    
                    # Print summary for this video
                    for person_id, frames in person_frames.items():
                        print(f"   ✓ {video_path.stem}/person_{person_id}: {len(frames)} frames")
                else:
                    failed += 1
                    print(f"   ✗ No valid persons detected in {video_path.name}")
                    
            except Exception as e:
                failed += 1
                print(f"   ✗ Error processing {video_path.name}: {e}")
        
        class_summary[class_name] = {
            "processed": processed,
            "failed": failed,
            "persons": persons_found
        }
        total_processed += processed
        total_failed += failed
        total_persons += persons_found
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    
    for class_name, stats in class_summary.items():
        print(f"{class_name:25s} | Videos: {stats['processed']:3d} | "
              f"Failed: {stats['failed']:3d} | Persons: {stats['persons']:3d}")
    
    print("-" * 60)
    print(f"{'TOTAL':25s} | Videos: {total_processed:3d} | "
          f"Failed: {total_failed:3d} | Persons: {total_persons:3d}")
    print("=" * 60)
    print(f"\n✓ Frames saved to: {config.BEHAVIOR_FRAMES_DIR}")
    print(f"✓ Average persons per video: {total_persons / max(total_processed, 1):.2f}")


if __name__ == "__main__":
    process_all_videos()
