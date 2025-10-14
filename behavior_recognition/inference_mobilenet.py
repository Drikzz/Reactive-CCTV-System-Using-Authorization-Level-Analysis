"""
MobileNetV2 Inference Script
Performs inference on images, videos, or webcam feed using a trained MobileNetV2 model.
"""

import os
import sys
from pathlib import Path
import argparse
import torch
import torch.nn as nn
from torchvision import transforms, models
import cv2
import numpy as np
from PIL import Image
import time

# Add parent directory to path for imports
BASE_DIR = Path(__file__).parent.parent
sys.path.append(str(BASE_DIR))


class MobileNetV2Inference:
    """
    MobileNetV2 inference class for image and video predictions.
    """
    
    def __init__(self, model_path, class_names=None, device=None):
        """
        Initialize the inference engine.
        
        Args:
            model_path (str): Path to the trained model checkpoint
            class_names (list): List of class names. If None, will use indices
            device (str): Device to use ('cuda' or 'cpu')
        """
        self.device = device if device else ('cuda' if torch.cuda.is_available() else 'cpu')
        self.model_path = model_path
        self.class_names = class_names
        
        # Load model
        self.model, self.num_classes = self._load_model()
        
        # Set class names if not provided
        if self.class_names is None:
            self.class_names = [f"Class_{i}" for i in range(self.num_classes)]
        
        # Define transform (same as validation/test in training)
        self.transform = transforms.Compose([
            transforms.Resize((224, 224)),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225])
        ])
        
        print(f"✓ Model loaded from: {model_path}")
        print(f"✓ Device: {self.device}")
        print(f"✓ Classes: {self.class_names}")
        print(f"✓ Number of classes: {self.num_classes}\n")
    
    def _load_model(self):
        """
        Load the trained model from checkpoint.
        
        Returns:
            model: Loaded PyTorch model
            num_classes: Number of output classes
        """
        # Load checkpoint
        checkpoint = torch.load(self.model_path, map_location=self.device)
        num_classes = checkpoint['num_classes']
        
        # Load class names from checkpoint if not provided
        if self.class_names is None:
            self.class_names = checkpoint.get('class_names', None)
        
        # Build model architecture
        model = models.mobilenet_v2(weights=None)  # Updated to use 'weights' parameter
        in_features = model.classifier[1].in_features
        model.classifier[1] = nn.Linear(in_features, num_classes)
        
        # Load weights
        model.load_state_dict(checkpoint['model_state_dict'])
        model = model.to(self.device)
        model.eval()
        
        return model, num_classes
    
    def preprocess_image(self, image):
        """
        Preprocess an image for inference.
        
        Args:
            image: PIL Image or numpy array (BGR from OpenCV)
            
        Returns:
            tensor: Preprocessed image tensor
        """
        # Convert OpenCV BGR to PIL RGB if needed
        if isinstance(image, np.ndarray):
            image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
            image = Image.fromarray(image)
        
        # Apply transforms
        image_tensor = self.transform(image)
        image_tensor = image_tensor.unsqueeze(0)  # Add batch dimension
        
        return image_tensor
    
    def predict(self, image, return_probabilities=False):
        """
        Predict class for a single image.
        
        Args:
            image: PIL Image or numpy array (BGR from OpenCV)
            return_probabilities (bool): Whether to return all class probabilities
            
        Returns:
            dict: Prediction results containing class_name, class_id, confidence, and optionally probabilities
        """
        # Preprocess
        image_tensor = self.preprocess_image(image)
        image_tensor = image_tensor.to(self.device)
        
        # Inference
        with torch.no_grad():
            outputs = self.model(image_tensor)
            probabilities = torch.nn.functional.softmax(outputs, dim=1)
            confidence, predicted = torch.max(probabilities, 1)
        
        # Get results
        class_id = predicted.item()
        class_name = self.class_names[class_id]
        confidence_score = confidence.item()
        
        result = {
            'class_name': class_name,
            'class_id': class_id,
            'confidence': confidence_score
        }
        
        if return_probabilities:
            result['probabilities'] = {
                self.class_names[i]: probabilities[0][i].item() 
                for i in range(self.num_classes)
            }
        
        return result
    
    def predict_image(self, image_path, display=True, save_output=None):
        """
        Predict class for a single image file.
        
        Args:
            image_path (str): Path to the image file
            display (bool): Whether to display the image with prediction
            save_output (str): Path to save annotated image (optional)
            
        Returns:
            dict: Prediction results
        """
        # Load image
        image = Image.open(image_path).convert('RGB')
        
        # Predict
        result = self.predict(image, return_probabilities=True)
        
        # Print results
        print("\n" + "="*50)
        print("IMAGE PREDICTION RESULTS")
        print("="*50)
        print(f"Image: {image_path}")
        print(f"Predicted Class: {result['class_name']}")
        print(f"Confidence: {result['confidence']:.4f} ({result['confidence']*100:.2f}%)")
        print("\nAll Class Probabilities:")
        for class_name, prob in result['probabilities'].items():
            print(f"  {class_name}: {prob:.4f} ({prob*100:.2f}%)")
        print("="*50 + "\n")
        
        # Load with OpenCV for display/saving
        cv_image = cv2.imread(str(image_path))
        annotated_image = self._annotate_frame(cv_image, result)
        
        # Save annotated image if requested
        if save_output:
            cv2.imwrite(save_output, annotated_image)
            print(f"✓ Annotated image saved to: {save_output}\n")
        
        # Display image with prediction
        if display:
            self._display_prediction(cv_image, result)
        
        return result
    
    def predict_video(self, video_path, frame_skip=5, display=True, save_output=None):
        """
        Predict classes for video frames.
        
        Args:
            video_path (str): Path to video file or camera index (0 for webcam)
            frame_skip (int): Process every Nth frame (default: 5)
            display (bool): Whether to display video with predictions
            save_output (str): Path to save output video (optional)
            
        Returns:
            list: List of predictions for each processed frame
        """
        # Open video
        if video_path == 'webcam' or (isinstance(video_path, int)):
            cap = cv2.VideoCapture(0)
            is_webcam = True
            print("✓ Webcam opened")
        else:
            cap = cv2.VideoCapture(str(video_path))
            is_webcam = False
            print(f"✓ Video opened: {video_path}")
        
        if not cap.isOpened():
            raise ValueError(f"Could not open video source: {video_path}")
        
        # Get video properties
        fps = int(cap.get(cv2.CAP_PROP_FPS))
        width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
        height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
        total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT)) if not is_webcam else None
        
        print(f"Video Info: {width}x{height} @ {fps} FPS")
        if total_frames:
            print(f"Total Frames: {total_frames}")
        print(f"Processing every {frame_skip} frame(s)\n")
        
        # Setup video writer if saving output
        out = None
        if save_output:
            fourcc = cv2.VideoWriter_fourcc(*'mp4v')
            out = cv2.VideoWriter(save_output, fourcc, fps, (width, height))
            print(f"✓ Saving output to: {save_output}\n")
        
        # Process video
        predictions = []
        frame_count = 0
        processed_count = 0
        
        print("Press 'q' to quit\n")
        
        try:
            while True:
                ret, frame = cap.read()
                
                if not ret:
                    break
                
                frame_count += 1
                
                # Process every Nth frame
                if frame_count % frame_skip == 0:
                    # Predict
                    start_time = time.time()
                    result = self.predict(frame)
                    inference_time = time.time() - start_time
                    
                    # Add metadata
                    result['frame_number'] = frame_count
                    result['inference_time'] = inference_time
                    predictions.append(result)
                    processed_count += 1
                    
                    # Display info
                    if not is_webcam and total_frames:
                        progress = (frame_count / total_frames) * 100
                        print(f"Frame {frame_count}/{total_frames} ({progress:.1f}%) - "
                              f"Predicted: {result['class_name']} "
                              f"(Conf: {result['confidence']:.2f}) - "
                              f"Time: {inference_time*1000:.1f}ms")
                    else:
                        print(f"Frame {frame_count} - "
                              f"Predicted: {result['class_name']} "
                              f"(Conf: {result['confidence']:.2f}) - "
                              f"FPS: {1/inference_time:.1f}")
                else:
                    # Use last prediction for non-processed frames
                    if predictions:
                        result = predictions[-1]
                    else:
                        result = {'class_name': 'Processing...', 'confidence': 0.0}
                
                # Annotate frame
                annotated_frame = self._annotate_frame(frame, result)
                
                # Display
                if display:
                    cv2.imshow('MobileNetV2 Inference', annotated_frame)
                    
                    # Check for quit
                    if cv2.waitKey(1) & 0xFF == ord('q'):
                        print("\nStopped by user")
                        break
                
                # Save frame
                if out:
                    out.write(annotated_frame)
        
        finally:
            # Cleanup
            cap.release()
            if out:
                out.release()
            if display:
                cv2.destroyAllWindows()
        
        # Print summary
        print("\n" + "="*50)
        print("VIDEO PREDICTION SUMMARY")
        print("="*50)
        print(f"Total Frames: {frame_count}")
        print(f"Processed Frames: {processed_count}")
        if predictions:
            avg_inference_time = np.mean([p['inference_time'] for p in predictions])
            print(f"Average Inference Time: {avg_inference_time*1000:.2f}ms")
            print(f"Average FPS: {1/avg_inference_time:.1f}")
        print("="*50 + "\n")
        
        return predictions
    
    def _annotate_frame(self, frame, result):
        """
        Annotate frame with prediction results.
        
        Args:
            frame: OpenCV frame (numpy array)
            result: Prediction result dictionary
            
        Returns:
            annotated_frame: Frame with annotations
        """
        annotated_frame = frame.copy()
        height, width = frame.shape[:2]
        
        # Create semi-transparent overlay for text background
        overlay = annotated_frame.copy()
        
        # Background rectangle
        cv2.rectangle(overlay, (10, 10), (width - 10, 120), (0, 0, 0), -1)
        cv2.addWeighted(overlay, 0.6, annotated_frame, 0.4, 0, annotated_frame)
        
        # Text configuration
        font = cv2.FONT_HERSHEY_SIMPLEX
        
        # Class name
        class_text = f"Class: {result['class_name']}"
        cv2.putText(annotated_frame, class_text, (20, 45), font, 1.0, (255, 255, 255), 2)
        
        # Confidence
        conf_text = f"Confidence: {result['confidence']:.2f} ({result['confidence']*100:.1f}%)"
        cv2.putText(annotated_frame, conf_text, (20, 85), font, 0.8, (0, 255, 0), 2)
        
        # Confidence bar
        bar_width = int((width - 40) * result['confidence'])
        cv2.rectangle(annotated_frame, (20, 100), (20 + bar_width, 110), (0, 255, 0), -1)
        cv2.rectangle(annotated_frame, (20, 100), (width - 20, 110), (255, 255, 255), 2)
        
        return annotated_frame
    
    def _display_prediction(self, image, result):
        """
        Display image with prediction overlay.
        
        Args:
            image: OpenCV image (numpy array)
            result: Prediction result dictionary
        """
        annotated_image = self._annotate_frame(image, result)
        
        # Resize for display if too large
        height, width = annotated_image.shape[:2]
        max_width = 1280
        if width > max_width:
            scale = max_width / width
            new_width = max_width
            new_height = int(height * scale)
            annotated_image = cv2.resize(annotated_image, (new_width, new_height))
        
        try:
            cv2.imshow('MobileNetV2 Prediction', annotated_image)
            print("Press any key to close...")
            cv2.waitKey(0)
            cv2.destroyAllWindows()
        except Exception as e:
            print(f"Note: Cannot display image (running in headless environment)")
            print(f"      Image saved would be available if --save option was used.")


def main():
    """
    Main function with command-line interface.
    """
    parser = argparse.ArgumentParser(description='MobileNetV2 Inference')
    
    # Mode selection (mutually exclusive)
    mode_group = parser.add_mutually_exclusive_group(required=True)
    mode_group.add_argument('--image', type=str, help='Path to input image')
    mode_group.add_argument('--video', type=str, help='Path to input video')
    mode_group.add_argument('--webcam', action='store_true', help='Use webcam input')
    
    # Model configuration
    parser.add_argument('--model', type=str, 
                       default='models/mobilenet/mobilenet_best.pth',
                       help='Path to trained model checkpoint (default: models/mobilenet/mobilenet_best.pth)')
    parser.add_argument('--classes', type=str, nargs='+',
                       help='Class names (optional, will auto-detect if not provided)')
    
    # Video-specific options
    parser.add_argument('--frame-skip', type=int, default=5,
                       help='Process every Nth frame for video/webcam (default: 5)')
    parser.add_argument('--no-display', action='store_true',
                       help='Do not display output window')
    parser.add_argument('--save', type=str,
                       help='Save output (video or annotated image) to specified path')
    
    # Device
    parser.add_argument('--device', type=str, choices=['cuda', 'cpu'],
                       help='Device to use (default: auto-detect)')
    
    args = parser.parse_args()
    
    # Resolve model path
    model_path = Path(args.model)
    if not model_path.is_absolute():
        model_path = BASE_DIR / model_path
    
    if not model_path.exists():
        print(f"Error: Model file not found: {model_path}")
        print("\nPlease train a model first using train_mobilenet.py")
        return
    
    # Initialize inference engine
    print("\n" + "="*50)
    print("MOBILENETV2 INFERENCE")
    print("="*50)
    
    inference = MobileNetV2Inference(
        model_path=str(model_path),
        class_names=args.classes,
        device=args.device
    )
    
    # Run inference based on mode
    if args.image:
        # Image mode
        image_path = Path(args.image)
        if not image_path.exists():
            print(f"Error: Image file not found: {image_path}")
            return
        
        inference.predict_image(
            image_path=str(image_path),
            display=not args.no_display,
            save_output=args.save
        )
    
    elif args.video:
        # Video mode
        video_path = Path(args.video)
        if not video_path.exists():
            print(f"Error: Video file not found: {video_path}")
            return
        
        inference.predict_video(
            video_path=str(video_path),
            frame_skip=args.frame_skip,
            display=not args.no_display,
            save_output=args.save
        )
    
    elif args.webcam:
        # Webcam mode
        inference.predict_video(
            video_path='webcam',
            frame_skip=args.frame_skip,
            display=not args.no_display,
            save_output=args.save
        )
    
    print("\nInference complete!")


if __name__ == "__main__":
    main()
