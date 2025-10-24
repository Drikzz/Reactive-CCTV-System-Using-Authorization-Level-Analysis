import os
import cv2
import numpy as np
from typing import Tuple

"""
Utility helpers for simple multi-angle capture:
- check_brightness
- check_blur
- save_image
"""

def check_brightness(image: np.ndarray, min_val: int = 50, max_val: int = 200) -> Tuple[bool, str]:
    """Return (ok, message)."""
    if image is None or image.size == 0:
        return False, "Image is empty"
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return False, f"Failed to convert to grayscale: {e}"
    mean_brightness = float(np.mean(gray))
    if mean_brightness < min_val:
        return False, "Image is too dark"
    if mean_brightness > max_val:
        return False, "Image is too bright"
    return True, "Brightness is acceptable"

def check_blur(image: np.ndarray, threshold: float = 100.0) -> Tuple[bool, str]:
    """Return (ok, message) based on Laplacian variance."""
    if image is None or image.size == 0:
        return False, "Image is empty"
    try:
        gray = cv2.cvtColor(image, cv2.COLOR_BGR2GRAY)
    except Exception as e:
        return False, f"Failed to convert to grayscale: {e}"
    try:
        variance = float(cv2.Laplacian(gray, cv2.CV_64F).var())
    except Exception as e:
        return False, f"Blur check failed: {e}"
    if variance < threshold:
        return False, "Image is too blurry"
    return True, "Image is sharp enough"

def save_image(image: np.ndarray, path: str) -> Tuple[bool, str]:
    """Save image to disk, creating parent dirs as needed."""
    try:
        os.makedirs(os.path.dirname(path), exist_ok=True)
        ok = cv2.imwrite(path, image)
        if not ok:
            return False, "cv2.imwrite returned False"
        return True, "Image saved successfully"
    except Exception as e:
        return False, f"Failed to save image: {e}"