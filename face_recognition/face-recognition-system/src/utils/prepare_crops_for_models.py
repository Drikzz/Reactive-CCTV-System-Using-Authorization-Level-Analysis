import cv2
import numpy as np

try:
    import torch
except Exception:
    torch = None

# Default target sizes (adjust if your models expect different)
DEFAULT_SIZES = {
    "facenet": 160,
    "arcface": 112,
    "dlib": 150
}

def _bgr_to_rgb(img):
    return cv2.cvtColor(img, cv2.COLOR_BGR2RGB)

def _resize(img, size):
    return cv2.resize(img, (size, size), interpolation=cv2.INTER_LINEAR)

def _preprocess_facenet(img_rgb):
    # FaceNet common preprocessing: float32, scale to [-1,1]
    img = img_rgb.astype(np.float32)
    return (img - 127.5) / 128.0

def _preprocess_arcface(img_rgb):
    # InsightFace / ArcFace common preprocessing: float32, scale to [-1,1]
    img = img_rgb.astype(np.float32)
    return (img - 127.5) / 128.0

def _preprocess_dlib(img_rgb):
    # Dlib typically uses RGB float in range [0,1] (or raw uint8). Use [0,1].
    img = img_rgb.astype(np.float32) / 255.0
    return img

def preprocess_crop_for_model(crop_bgr, model_name="facenet", as_tensor=False, device="cpu"):
    """
    Prepare a single crop for a specific model.

    Args:
      crop_bgr: np.ndarray BGR image (OpenCV read crop)
      model_name: "facenet" | "arcface" | "dlib" (case-insensitive)
      as_tensor: if True and torch available, returns a torch.Tensor shaped (1,C,H,W)
      device: torch device string when as_tensor=True

    Returns:
      numpy array (H,W,C) float32 normalized OR torch.Tensor (1,C,H,W)
    """
    if crop_bgr is None:
        return None

    m = model_name.lower()
    size = DEFAULT_SIZES.get(m, DEFAULT_SIZES["facenet"])

    # convert to RGB and resize
    img_rgb = _bgr_to_rgb(crop_bgr)
    img_rgb = _resize(img_rgb, size)

    # model-specific normalization
    if m == "facenet":
        proc = _preprocess_facenet(img_rgb)
    elif m == "arcface":
        proc = _preprocess_arcface(img_rgb)
    elif m == "dlib":
        proc = _preprocess_dlib(img_rgb)
    else:
        # safe default: scale to [-1,1]
        proc = (img_rgb.astype(np.float32) - 127.5) / 128.0

    if as_tensor and torch is not None:
        # convert HWC -> CHW and add batch dim
        t = torch.from_numpy(proc.transpose(2, 0, 1)).unsqueeze(0).to(device)
        return t
    return proc

def preprocess_folder_of_crops(crops_dir, model_names=("facenet","arcface","dlib"), limit=0, as_tensor=False, device="cpu"):
    """
    Batch preprocess all images in a folder.

    Returns dict: { filename: { model_name: processed_array_or_tensor } }
    """
    from pathlib import Path
    p = Path(crops_dir)
    imgs = sorted([x for x in p.iterdir() if x.suffix.lower() in (".jpg",".jpeg",".png")])
    if limit and limit > 0:
        imgs = imgs[:limit]

    out = {}
    for img_path in imgs:
        try:
            img = cv2.imread(str(img_path))
            if img is None:
                continue
            out[img_path.name] = {}
            for m in model_names:
                out[img_path.name][m] = preprocess_crop_for_model(img, model_name=m, as_tensor=as_tensor, device=device)
        except Exception:
            continue
    return out