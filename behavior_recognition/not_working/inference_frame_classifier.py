"""Inference script for frame-based MobileNetV2 classifier.

This script performs inference on individual frames using a trained MobileNetV2 classifier.
Supports both single image and batch processing modes.
"""
from __future__ import annotations
import os
import sys
import json
import time
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np
import torch
import torch.nn as nn
import torchvision.transforms as T
from PIL import Image
import cv2
from torchvision.models import mobilenet_v2


# -------------------------- Model Definition -------------------------- #

class MobileNetV2Classifier(nn.Module):
    """MobileNetV2 backbone with custom classifier head for frame classification."""
    
    def __init__(
        self,
        num_classes: int,
        pretrained: bool = True,
        freeze_backbone: bool = False,
        dropout: float = 0.3
    ):
        """
        Args:
            num_classes: Number of output classes
            pretrained: Whether to use pretrained ImageNet weights
            freeze_backbone: Whether to freeze backbone (only train classifier)
            dropout: Dropout probability in classifier
        """
        super().__init__()
        
        # Load MobileNetV2 backbone
        self.backbone = mobilenet_v2(pretrained=pretrained)
        
        # Freeze backbone if requested
        if freeze_backbone:
            for param in self.backbone.parameters():
                param.requires_grad = False
        
        # Replace classifier head
        # MobileNetV2 feature dimension is 1280
        in_features = 1280
        
        self.backbone.classifier = nn.Sequential(
            nn.Dropout(p=dropout),
            nn.Linear(in_features, num_classes)
        )
    
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        """
        Args:
            x: Input tensor of shape (B, 3, H, W)
        
        Returns:
            Logits of shape (B, num_classes)
        """
        return self.backbone(x)


# -------------------------- Inference Engine -------------------------- #

class FrameClassifierInference:
    """Inference wrapper for frame-based MobileNetV2 classifier."""
    
    def __init__(
        self,
        model_dir: str = 'models/frame_classifier_mobilenet',
        device: Optional[torch.device] = None
    ):
        """
        Initialize inference engine.
        
        Args:
            model_dir: Directory containing model checkpoint and metadata
            device: Device to run inference on (auto-detect if None)
        """
        self.model_dir = model_dir
        self.device = device if device is not None else \
            torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        
        # Load config
        config_path = os.path.join(model_dir, 'config.json')
        if not os.path.isfile(config_path):
            raise FileNotFoundError(f"Config not found: {config_path}")
        
        with open(config_path, 'r', encoding='utf-8') as f:
            self.config = json.load(f)
        
        # Load classes
        classes_path = os.path.join(model_dir, 'classes.json')
        if not os.path.isfile(classes_path):
            raise FileNotFoundError(f"Classes file not found: {classes_path}")
        
        with open(classes_path, 'r', encoding='utf-8') as f:
            self.classes = json.load(f)
        
        self.num_classes = len(self.classes)
        self.img_size = self.config.get('img_size', 224)
        
        # Build model
        self.model = MobileNetV2Classifier(
            num_classes=self.num_classes,
            pretrained=False,  # We're loading trained weights
            freeze_backbone=False,
            dropout=self.config.get('dropout', 0.3)
        ).to(self.device)
        
        # Load weights
        model_path = os.path.join(model_dir, 'best_model.pth')
        if not os.path.isfile(model_path):
            raise FileNotFoundError(f"Model checkpoint not found: {model_path}")
        
        self.model.load_state_dict(torch.load(model_path, map_location=self.device, weights_only=True))
        self.model.eval()
        
        # Preprocessing transform
        self.transform = T.Compose([
            T.Resize((self.img_size, self.img_size)),
            T.ToTensor(),
            T.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Classes ({self.num_classes}): {self.classes}")
    
    def preprocess_image(self, image_input) -> torch.Tensor:
        """
        Preprocess image for inference.
        
        Args:
            image_input: Can be:
                - str: Path to image file
                - PIL.Image.Image: PIL image
                - np.ndarray: OpenCV/numpy image (BGR or RGB)
        
        Returns:
            Preprocessed tensor of shape (1, 3, H, W)
        """
        if isinstance(image_input, str):
            # Load from file
            img = Image.open(image_input).convert('RGB')
        elif isinstance(image_input, Image.Image):
            # Already PIL image
            img = image_input.convert('RGB')
        elif isinstance(image_input, np.ndarray):
            # Convert numpy/OpenCV to PIL
            # Assume BGR (OpenCV default), convert to RGB
            if len(image_input.shape) == 3 and image_input.shape[2] == 3:
                img = Image.fromarray(cv2.cvtColor(image_input, cv2.COLOR_BGR2RGB))
            else:
                img = Image.fromarray(image_input)
        else:
            raise TypeError(f"Unsupported image input type: {type(image_input)}")
        
        # Apply transform and add batch dimension
        tensor = self.transform(img)
        return tensor.unsqueeze(0)
    
    def predict(
        self,
        image_input,
        top_k: int = 1,
        return_probabilities: bool = False
    ) -> Tuple[List[str], List[float]]:
        """
        Predict class for a single image.
        
        Args:
            image_input: Image to classify (see preprocess_image for formats)
            top_k: Number of top predictions to return
            return_probabilities: Whether to return probabilities or raw logits
        
        Returns:
            classes: List of predicted class names (top-k)
            scores: List of confidence scores (top-k)
        """
        # Preprocess
        tensor = self.preprocess_image(image_input).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(tensor)
            
            if return_probabilities:
                probs = torch.softmax(logits, dim=1)
                scores, indices = torch.topk(probs, k=min(top_k, self.num_classes), dim=1)
            else:
                scores, indices = torch.topk(logits, k=min(top_k, self.num_classes), dim=1)
        
        # Convert to lists
        predicted_classes = [self.classes[idx] for idx in indices[0].cpu().tolist()]
        confidence_scores = scores[0].cpu().tolist()
        
        return predicted_classes, confidence_scores
    
    def predict_batch(
        self,
        image_inputs: List,
        return_probabilities: bool = False
    ) -> Tuple[List[str], List[float]]:
        """
        Predict classes for a batch of images.
        
        Args:
            image_inputs: List of images (see preprocess_image for formats)
            return_probabilities: Whether to return probabilities or raw logits
        
        Returns:
            classes: List of predicted class names
            scores: List of confidence scores
        """
        # Preprocess all images
        tensors = [self.preprocess_image(img) for img in image_inputs]
        batch = torch.cat(tensors, dim=0).to(self.device)
        
        # Inference
        with torch.no_grad():
            logits = self.model(batch)
            
            if return_probabilities:
                probs = torch.softmax(logits, dim=1)
                scores, indices = torch.max(probs, dim=1)
            else:
                scores, indices = torch.max(logits, dim=1)
        
        # Convert to lists
        predicted_classes = [self.classes[idx] for idx in indices.cpu().tolist()]
        confidence_scores = scores.cpu().tolist()
        
        return predicted_classes, confidence_scores
    
    def predict_video(
        self,
        video_path: str,
        sample_rate: int = 5,
        return_probabilities: bool = True,
        display: bool = False
    ) -> Tuple[List[str], List[float], List[int]]:
        """
        Predict on video by sampling frames.
        
        Args:
            video_path: Path to video file
            sample_rate: Sample every Nth frame
            return_probabilities: Whether to return probabilities
            display: Whether to display video with predictions in real-time
        
        Returns:
            classes: List of predicted class names for each sampled frame
            scores: List of confidence scores
            frame_indices: List of frame indices that were sampled
        """
        cap = cv2.VideoCapture(video_path)
        if not cap.isOpened():
            raise ValueError(f"Failed to open video: {video_path}")
        
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
        fps = cap.get(cv2.CAP_PROP_FPS)
        
        print(f"Processing video: {video_path}")
        print(f"  Total frames: {total_frames}")
        print(f"  FPS: {fps:.2f}")
        print(f"  Sample rate: 1/{sample_rate} frames")
        if display:
            print("  Display: ON (Press 'q' to quit)")
        
        predictions = []
        confidences = []
        frame_indices = []
        sampled_frames = []
        
        frame_idx = 0
        try:
            while True:
                ret, frame = cap.read()
                if not ret:
                    break
                
                if frame_idx % sample_rate == 0:
                    # Predict on this frame
                    pred_classes, pred_scores = self.predict(
                        frame,
                        top_k=1,
                        return_probabilities=return_probabilities
                    )
                    
                    predictions.append(pred_classes[0])
                    confidences.append(pred_scores[0])
                    frame_indices.append(frame_idx)
                    
                    if display:
                        sampled_frames.append(frame.copy())
                
                frame_idx += 1
            
            cap.release()
            
            print(f"Processed {len(frame_indices)} frames")
            
            # Display frames with predictions if requested
            if display and sampled_frames:
                print("\nDisplaying predictions (Press 'q' to skip)...")
                for i, frame in enumerate(sampled_frames):
                    display_frame = frame.copy()
                    
                    # Add prediction text (large at top)
                    text = f"{predictions[i]}: {confidences[i]:.2f}"
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    
                    # Add frame info (smaller at bottom)
                    info_text = f"Frame: {frame_indices[i]}/{total_frames} ({i+1}/{len(sampled_frames)} sampled)"
                    cv2.putText(
                        display_frame,
                        info_text,
                        (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.5,
                        (255, 255, 255),
                        1
                    )
                    
                    cv2.imshow("Video Frame Classifier", display_frame)
                    
                    # Wait for key press (1ms for auto-play, or until 'q' is pressed)
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
                
                cv2.destroyAllWindows()
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        
        return predictions, confidences, frame_indices
    
    def predict_webcam(
        self,
        camera_id: int = 0,
        display: bool = True,
        return_probabilities: bool = True
    ):
        """
        Real-time prediction on webcam feed.
        
        Args:
            camera_id: Camera device ID
            display: Whether to display visualization
            return_probabilities: Whether to use probabilities
        """
        cap = cv2.VideoCapture(camera_id)
        if not cap.isOpened():
            raise ValueError(f"Failed to open camera {camera_id}")
        
        print("Starting webcam inference...")
        print("Press 'q' to quit")
        
        fps_counter = []
        
        try:
            while True:
                start_time = time.time()
                
                ret, frame = cap.read()
                if not ret:
                    break
                
                # Predict
                pred_classes, pred_scores = self.predict(
                    frame,
                    top_k=3,
                    return_probabilities=return_probabilities
                )
                
                # Calculate FPS
                elapsed = time.time() - start_time
                fps_counter.append(1.0 / max(elapsed, 1e-6))
                if len(fps_counter) > 30:
                    fps_counter.pop(0)
                avg_fps = np.mean(fps_counter)
                
                if display:
                    # Draw predictions on frame
                    display_frame = frame.copy()
                    
                    # Top prediction (large text)
                    text = f"{pred_classes[0]}: {pred_scores[0]:.2f}"
                    cv2.putText(
                        display_frame,
                        text,
                        (10, 40),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        1.0,
                        (0, 255, 0),
                        2
                    )
                    
                    # Top-3 predictions (smaller text)
                    y_offset = 80
                    for i, (cls, score) in enumerate(zip(pred_classes[:3], pred_scores[:3])):
                        text = f"{i+1}. {cls}: {score:.2f}"
                        cv2.putText(
                            display_frame,
                            text,
                            (10, y_offset + i * 30),
                            cv2.FONT_HERSHEY_SIMPLEX,
                            0.6,
                            (255, 255, 255),
                            1
                        )
                    
                    # FPS
                    cv2.putText(
                        display_frame,
                        f"FPS: {avg_fps:.1f}",
                        (10, display_frame.shape[0] - 10),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.6,
                        (255, 255, 0),
                        1
                    )
                    
                    cv2.imshow('Frame Classifier', display_frame)
                    
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        break
        
        finally:
            cap.release()
            if display:
                cv2.destroyAllWindows()
        
        print("Webcam inference stopped")


def main():
    """Example usage."""
    import argparse
    
    parser = argparse.ArgumentParser(description='Frame-based classifier inference')
    parser.add_argument('--model_dir', type=str, default='models/frame_classifier_mobilenet',
                        help='Model directory')
    parser.add_argument('--mode', type=str, choices=['image', 'video', 'webcam'], default='webcam',
                        help='Inference mode')
    parser.add_argument('--input', type=str, help='Input image or video path')
    parser.add_argument('--camera_id', type=int, default=0, help='Camera ID for webcam mode')
    parser.add_argument('--sample_rate', type=int, default=5, help='Sample every Nth frame (video mode)')
    parser.add_argument('--top_k', type=int, default=3, help='Number of top predictions (image mode)')
    parser.add_argument('--display', action='store_true', help='Display video with predictions (video mode)')
    
    args = parser.parse_args()
    
    # Initialize inference engine
    inferencer = FrameClassifierInference(model_dir=args.model_dir)
    
    if args.mode == 'image':
        if not args.input:
            print("Error: --input required for image mode")
            return
        
        print(f"\nPredicting on image: {args.input}")
        classes, scores = inferencer.predict(
            args.input,
            top_k=args.top_k,
            return_probabilities=True
        )
        
        print(f"\nTop-{args.top_k} predictions:")
        for i, (cls, score) in enumerate(zip(classes, scores)):
            print(f"  {i+1}. {cls}: {score:.4f} ({score*100:.2f}%)")
    
    elif args.mode == 'video':
        if not args.input:
            print("Error: --input required for video mode")
            return
        
        print(f"\nPredicting on video: {args.input}")
        classes, scores, frame_indices = inferencer.predict_video(
            args.input,
            sample_rate=args.sample_rate,
            return_probabilities=True,
            display=args.display
        )
        
        # Show summary
        from collections import Counter
        class_counts = Counter(classes)
        
        print("\nPrediction summary:")
        for cls, count in class_counts.most_common():
            percentage = 100 * count / len(classes)
            print(f"  {cls}: {count}/{len(classes)} frames ({percentage:.1f}%)")
    
    elif args.mode == 'webcam':
        print("\nStarting webcam inference...")
        inferencer.predict_webcam(
            camera_id=args.camera_id,
            display=True,
            return_probabilities=True
        )


if __name__ == '__main__':
    main()
