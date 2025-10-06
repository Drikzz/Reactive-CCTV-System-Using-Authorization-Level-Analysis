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


def extract_keypoints_from_video(video_path, model, conf_threshold=0.3):
    """
    Extract normalized keypoints from a video file.
    
    Args:
        video_path: Path to input video
        model: YOLOv8-pose model
        conf_threshold: Confidence threshold for detections
        
    Returns:
        torch.Tensor of shape [num_frames, 17, 2] containing normalized (x, y) coordinates
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
            
            # Extract keypoints from the first detected person
            if results[0].keypoints is not None and len(results[0].keypoints.xy) > 0:
                keypoints = results[0].keypoints.xy.cpu().numpy()  # Shape: [num_persons, 17, 2]
                
                # Take first person's keypoints and normalize
                kp = keypoints[0].copy()  # Shape: [17, 2]
                kp[:, 0] = kp[:, 0] / frame_width   # Normalize x
                kp[:, 1] = kp[:, 1] / frame_height  # Normalize y
                keypoints_sequence.append(kp)
            else:
                # No detection, add zeros
                keypoints_sequence.append(np.zeros((17, 2)))
            
            pbar.update(1)
    
    cap.release()
    
    if len(keypoints_sequence) == 0:
        print(f"Warning: No frames processed for {video_path}")
        return None
    
    # Convert to tensor: [num_frames, 17, 2]
    keypoints_tensor = torch.tensor(np.array(keypoints_sequence), dtype=torch.float32)
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
    
    # Check if input directory exists
    if not config.BEHAVIOR_CLIPS_DIR.exists():
        print(f"\n❌ Error: Input directory not found: {config.BEHAVIOR_CLIPS_DIR}")
        print("Please create the directory and add video files organized by class.")
        return
    
    # Get all class directories
    class_dirs = [d for d in config.BEHAVIOR_CLIPS_DIR.iterdir() if d.is_dir()]
    
    if len(class_dirs) == 0:
        print(f"\n❌ No class directories found in {config.BEHAVIOR_CLIPS_DIR}")
        print("Expected structure: behavior_clips/suspicious-actions/<class_name>/*.mp4")
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
                    conf_threshold=config.YOLO_CONF_THRESHOLD
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
