"""
Step 1: Extract Pose Sequences from Behavior Videos
Extracts 17 keypoints from each video frame using YOLOv8-pose and saves as PyTorch tensors.
"""
import cv2
import torch
import numpy as np
from pathlib import Path
from tqdm import tqdm
from ultralytics import YOLO
import config


def select_best_person(keypoints_data, boxes_data, strategy='largest'):
    """
    Select the best person from multiple detections.
    
    Args:
        keypoints_data: numpy array of shape [num_persons, 17, 2]
        boxes_data: numpy array of shape [num_persons, 4] (x1, y1, x2, y2)
        strategy: 'largest' (biggest bbox = closest) or 'first'
        
    Returns:
        Selected keypoints of shape [17, 2]
    """
    if len(keypoints_data) == 0:
        return None
    
    if strategy == 'largest' and boxes_data is not None and len(boxes_data) > 0:
        # Calculate bounding box areas (larger = closer to camera)
        areas = (boxes_data[:, 2] - boxes_data[:, 0]) * (boxes_data[:, 3] - boxes_data[:, 1])
        best_idx = np.argmax(areas)
        return keypoints_data[best_idx]
    else:
        # Default to first detection
        return keypoints_data[0]


def resample_sequence(sequence, target_length):
    """
    Resample sequence to fixed length by uniform sampling.
    
    Args:
        sequence: numpy array of shape [num_frames, 17, 2]
        target_length: desired number of frames
        
    Returns:
        Resampled sequence of shape [target_length, 17, 2]
    """
    if len(sequence) == 0:
        return np.zeros((target_length, 17, 2))
    
    if len(sequence) == target_length:
        return sequence
    
    # Uniformly sample indices
    indices = np.linspace(0, len(sequence) - 1, target_length, dtype=int)
    return sequence[indices]


def extract_keypoints_from_video(video_path, model, conf_threshold=0.3, 
                                  fixed_length=None, person_strategy='largest'):
    """
    Extract normalized keypoints from a video file.
    
    Args:
        video_path: Path to input video
        model: YOLOv8-pose model
        conf_threshold: Confidence threshold for detections
        fixed_length: If set, resample all videos to this many frames
        person_strategy: 'largest' (closest to camera) or 'first' for multi-person videos
        
    Returns:
        torch.Tensor of shape [num_frames, 17, 2] or [fixed_length, 17, 2] 
        containing normalized (x, y) coordinates
    """
    cap = cv2.VideoCapture(str(video_path))
    
    if not cap.isOpened():
        print(f"Error: Cannot open video {video_path}")
        return None
    
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    keypoints_sequence = []
    
    with tqdm(total=total_frames, desc=f"Processing {video_path.name}", leave=False) as pbar:
        while True:
            ret, frame = cap.read()
            if not ret:
                break
            
            # Run YOLOv8-pose prediction
            results = model(frame, conf=conf_threshold, verbose=False)
            
            # Extract keypoints from the best detected person
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy.cpu().numpy()  # Shape: [num_persons, 17, 2]
                
                # Get bounding boxes for multi-person selection
                boxes = None
                if results[0].boxes is not None and len(results[0].boxes.xyxy) > 0:
                    boxes = results[0].boxes.xyxy.cpu().numpy()  # Shape: [num_persons, 4]
                
                # Select best person (largest bbox = closest to camera)
                kp = select_best_person(keypoints, boxes, strategy=person_strategy)
                
                if kp is not None:
                    # Normalize keypoints
                    kp = kp.copy()
                    kp[:, 0] = kp[:, 0] / frame_width   # Normalize x
                    kp[:, 1] = kp[:, 1] / frame_height  # Normalize y
                    keypoints_sequence.append(kp)
                else:
                    # No valid detection, add zeros
                    keypoints_sequence.append(np.zeros((17, 2)))
            else:
                # No detection, add zeros
                keypoints_sequence.append(np.zeros((17, 2)))
            
            pbar.update(1)
    
    cap.release()
    
    if len(keypoints_sequence) == 0:
        print(f"Warning: No frames processed for {video_path}")
        return None
    
    # Convert to numpy array
    keypoints_array = np.array(keypoints_sequence)
    
    # Resample to fixed length if specified
    if fixed_length is not None and fixed_length > 0:
        keypoints_array = resample_sequence(keypoints_array, fixed_length)
    
    # Convert to tensor: [num_frames, 17, 2] or [fixed_length, 17, 2]
    keypoints_tensor = torch.tensor(keypoints_array, dtype=torch.float32)
    return keypoints_tensor


def process_all_videos():
    """
    Process all videos in behavior_clips directory and save pose sequences.
    """
    print("=" * 60)
    print("POSE SEQUENCE EXTRACTION PIPELINE")
    print("=" * 60)
    
    # Setup device
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    print(f"\n🖥️  Using device: {device}")
    if torch.cuda.is_available():
        print(f"   GPU: {torch.cuda.get_device_name(0)}")
    
    # Load YOLOv8-pose model
    print(f"\n📦 Loading YOLOv8-pose model from {config.YOLO_POSE_MODEL}")
    model = YOLO(str(config.YOLO_POSE_MODEL))
    model.to(device)  # Move model to GPU if available
    print("✓ Model loaded successfully")
    
    # Display extraction settings
    print(f"\n⚙️  Extraction Settings:")
    print(f"   Confidence threshold: {config.YOLO_CONF_THRESHOLD}")
    print(f"   Multi-person strategy: {config.MULTI_PERSON_STRATEGY}")
    if config.FIXED_SEQUENCE_LENGTH:
        print(f"   Fixed sequence length: {config.FIXED_SEQUENCE_LENGTH} frames")
    else:
        print(f"   Sequence length: Variable (original video length)")
    
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
            class_dirs.extend(dataset_class_dirs)
            print(f"   Found {len(dataset_class_dirs)} classes in {dataset_name}")
        else:
            print(f"⚠️  Warning: Dataset directory not found: {dataset_path}")
    
    if len(class_dirs) == 0:
        print(f"\n❌ No class directories found in any dataset")
        print("Expected structure: behavior_clips/suspicious-actions/<class_name>/*.mp4")
        print("behavior_clips/neutral-actions/<class_name>/*.mp4")
        return
    
    print(f"\n📁 Found {len(class_dirs)} behavior classes")
    
    total_processed = 0
    total_failed = 0
    class_summary = {}
    
    # Process each class
    for class_dir in class_dirs:
        class_name = class_dir.name
        print(f"\n{'=' * 60}")
        print(f"Processing class: {class_name}")
        print(f"{'=' * 60}")
        
        # Get all video files
        video_files = list(class_dir.glob("*.mp4")) + list(class_dir.glob("*.avi"))
        
        if len(video_files) == 0:
            print(f"⚠️  No video files found in {class_dir}")
            continue
        
        print(f"Found {len(video_files)} videos")
        
        # Create output directory for this class
        output_dir = config.POSE_CACHE_DIR / class_name
        output_dir.mkdir(parents=True, exist_ok=True)
        
        processed = 0
        failed = 0
        
        # Process each video
        for video_path in video_files:
            try:
                # Extract keypoints
                keypoints_tensor = extract_keypoints_from_video(
                    video_path, 
                    model, 
                    conf_threshold=config.YOLO_CONF_THRESHOLD,
                    fixed_length=config.FIXED_SEQUENCE_LENGTH,
                    person_strategy=config.MULTI_PERSON_STRATEGY
                )
                
                if keypoints_tensor is not None:
                    # Save tensor
                    output_path = output_dir / f"{video_path.stem}.pt"
                    torch.save(keypoints_tensor, output_path)
                    processed += 1
                    print(f"✓ Saved: {output_path.name} [shape: {keypoints_tensor.shape}]")
                else:
                    failed += 1
                    print(f"✗ Failed to process: {video_path.name}")
                    
            except Exception as e:
                failed += 1
                print(f"✗ Error processing {video_path.name}: {e}")
        
        class_summary[class_name] = {"processed": processed, "failed": failed}
        total_processed += processed
        total_failed += failed
    
    # Print final summary
    print("\n" + "=" * 60)
    print("EXTRACTION SUMMARY")
    print("=" * 60)
    
    for class_name, stats in class_summary.items():
        print(f"{class_name:20s} | Processed: {stats['processed']:3d} | Failed: {stats['failed']:3d}")
    
    print("-" * 60)
    print(f"{'TOTAL':20s} | Processed: {total_processed:3d} | Failed: {total_failed:3d}")
    print("=" * 60)
    print(f"\n✓ Pose sequences saved to: {config.POSE_CACHE_DIR}")
    

if __name__ == "__main__":
    process_all_videos()
