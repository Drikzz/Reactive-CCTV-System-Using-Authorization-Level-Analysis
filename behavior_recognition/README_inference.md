# MobileNetV2 Inference Guide

## Recent Updates (Fixed Issues)

### ✅ Fixed Issues:
1. **Deprecated Warning**: Updated from `pretrained=True` to `weights=` parameter
2. **Headless Environment**: Added graceful handling for environments without display (Google Colab, servers)
3. **Class Names**: Model now automatically saves and loads actual class names (e.g., "opening-cabinet" instead of "Class_0")
4. **Image Save Option**: Added ability to save annotated images with `--save` parameter

## Usage Examples

### 1. Single Image Inference

#### Basic usage (with display if available):
```bash
python inference_mobilenet.py --image path/to/image.jpg
```

#### Headless mode (no display, useful for Colab/servers):
```bash
python inference_mobilenet.py --image path/to/image.jpg --no-display
```

#### Save annotated image (for headless environments):
```bash
python inference_mobilenet.py --image path/to/image.jpg --no-display --save output.jpg
```

### 2. Video File Inference

#### Basic usage:
```bash
python inference_mobilenet.py --video path/to/video.mp4
```

#### Process every 10 frames (faster):
```bash
python inference_mobilenet.py --video path/to/video.mp4 --frame-skip 10
```

#### Save annotated output video:
```bash
python inference_mobilenet.py --video path/to/video.mp4 --save output_annotated.mp4
```

### 3. Webcam Inference

#### Basic usage:
```bash
python inference_mobilenet.py --webcam
```

#### Save webcam recording with predictions:
```bash
python inference_mobilenet.py --webcam --save webcam_output.mp4
```

### 4. Advanced Options

#### Specify custom model path:
```bash
python inference_mobilenet.py --image test.jpg --model custom/path/model.pth
```

#### Manually specify class names (overrides model):
```bash
python inference_mobilenet.py --image test.jpg --classes opening-cabinet using-computer
```

#### Force CPU usage:
```bash
python inference_mobilenet.py --image test.jpg --device cpu
```

## For Google Colab / Jupyter Notebooks

Since Colab doesn't support GUI display, use these commands:

```python
# Image inference (no display, save output)
!python inference_mobilenet.py --image ../vid1frame.png --no-display --save annotated_vid1.jpg

# Display the saved image using Colab's image display
from IPython.display import Image, display
display(Image('annotated_vid1.jpg'))
```

Or programmatically:

```python
from inference_mobilenet import MobileNetV2Inference
from PIL import Image
import matplotlib.pyplot as plt

# Initialize inference
inference = MobileNetV2Inference(
    model_path='models/mobilenet/mobilenet_best.pth',
    device='cuda'  # or 'cpu'
)

# Predict on image (no display)
result = inference.predict_image(
    image_path='../vid1frame.png',
    display=False,
    save_output='annotated_output.jpg'
)

# Display using matplotlib
img = Image.open('annotated_output.jpg')
plt.figure(figsize=(10, 8))
plt.imshow(img)
plt.axis('off')
plt.title(f"Prediction: {result['class_name']} ({result['confidence']:.2%})")
plt.show()
```

## Expected Output

### Console Output:
```
==================================================
MOBILENETV2 INFERENCE
==================================================
✓ Model loaded from: models/mobilenet/mobilenet_best.pth
✓ Device: cuda
✓ Classes: ['opening-cabinet', 'using-computer']
✓ Number of classes: 2

==================================================
IMAGE PREDICTION RESULTS
==================================================
Image: ../vid1frame.png
Predicted Class: opening-cabinet
Confidence: 1.0000 (100.00%)

All Class Probabilities:
  opening-cabinet: 1.0000 (100.00%)
  using-computer: 0.0000 (0.00%)
==================================================

✓ Annotated image saved to: output.jpg
Note: Cannot display image (running in headless environment)
      Image saved would be available if --save option was used.
```

### Visual Output:
The annotated image/video will show:
- Black semi-transparent overlay at the top
- White text: "Class: [predicted_class]"
- Green text: "Confidence: [score] ([percentage]%)"
- Green progress bar showing confidence level

## Troubleshooting

### Issue: "Class_0, Class_1" instead of actual names

**Solution**: Retrain your model with the updated `train_mobilenet.py` script. The new version saves class names in the checkpoint.

### Issue: Qt/Display errors in headless environment

**Solution**: Use `--no-display` flag:
```bash
python inference_mobilenet.py --image test.jpg --no-display --save output.jpg
```

### Issue: Deprecation warnings

**Solution**: Both training and inference scripts have been updated to use the new `weights=` parameter instead of `pretrained=`.

## File Structure

```
behavior_recognition/
├── train_mobilenet.py          # Training script (saves class names)
├── inference_mobilenet.py      # Inference script (loads class names)
└── README_inference.md         # This file

models/
└── mobilenet/
    └── mobilenet_best.pth      # Trained model (contains class names)
```

## Next Steps

1. **Retrain your model** with the updated `train_mobilenet.py` to include class names
2. **Test inference** with the updated script
3. Use `--no-display --save` for headless environments
