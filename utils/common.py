import os
import cv2
import numpy as np
import math
import torch
from facenet_pytorch import fixed_image_standardization
from .facenet_utils import get_detector, get_embedder

def compute_blur_score(img_bgr):
	# Variance of Laplacian: higher is sharper
	return float(cv2.Laplacian(img_bgr, cv2.CV_64F).var())

def estimate_brightness(img_bgr):
	# Use HSV V-channel mean as brightness
	hsv = cv2.cvtColor(img_bgr, cv2.COLOR_BGR2HSV)
	return float(np.mean(hsv[..., 2]))

def eye_angle_deg(keypoints):
	try:
		lp = keypoints.get('left_eye')
		rp = keypoints.get('right_eye')
		if lp is None or rp is None:
			return 0.0
		dy = rp[1] - lp[1]
		dx = rp[0] - lp[0]
		return math.degrees(math.atan2(dy, dx))
	except Exception:
		return 0.0

def align_face(frame_bgr, box, keypoints, output_size=(160, 160)):
	"""
	Align face crop based on eye angle. Falls back to resized crop if keypoints are missing.
	box: (x, y, w, h) in original frame coords
	"""
	x, y, w, h = box
	x = max(0, x); y = max(0, y)
	w = max(1, min(frame_bgr.shape[1] - x, w))
	h = max(1, min(frame_bgr.shape[0] - y, h))
	face = frame_bgr[y:y+h, x:x+w]
	if face.size == 0:
		return None

	# Compute rotation angle using eyes relative to the crop
	angle = 0.0
	try:
		lp = keypoints.get('left_eye')
		rp = keypoints.get('right_eye')
		if lp is not None and rp is not None:
			lp_rel = (lp[0] - x, lp[1] - y)
			rp_rel = (rp[0] - x, rp[1] - y)
			dy = rp_rel[1] - lp_rel[1]
			dx = rp_rel[0] - lp_rel[0]
			angle = math.degrees(math.atan2(dy, dx))
	except Exception:
		angle = 0.0

	center = (w // 2, h // 2)
	M = cv2.getRotationMatrix2D(center, angle, 1.0)
	rotated = cv2.warpAffine(face, M, (w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REPLICATE)
	return cv2.resize(rotated, output_size)

def load_known_faces(folder="datasets/faces"):
	"""
	PyTorch version:
	- If .npy embeddings are present, load them (note: may be incompatible with TF-era .npy).
	- Otherwise, detect faces with MTCNN, align (160x160), embed with InceptionResnetV1.
	"""
	encodings, names = [], []
	if not os.path.isdir(folder):
		print(f"[WARN] Faces folder not found: {folder}")
		return encodings, names

	detector = get_detector()
	embedder = get_embedder()
	device = next(embedder.parameters()).device

	for person_name in os.listdir(folder):
		person_dir = os.path.join(folder, person_name)
		if not os.path.isdir(person_dir):
			continue

		# Prefer precomputed embeddings (if they were created with the same pipeline)
		npy_files = [f for f in os.listdir(person_dir) if f.lower().endswith(".npy")]
		for f in npy_files:
			try:
				emb = np.load(os.path.join(person_dir, f))
				if emb is not None and emb.size > 0:
					encodings.append(np.asarray(emb, dtype=np.float32))
					names.append(person_name)
			except Exception as e:
				print(f"[WARN] Failed to load embedding {f}: {e}")

		# Fallback: compute from images
		if not npy_files:
			img_files = [f for f in os.listdir(person_dir) if f.lower().endswith((".png", ".jpg", ".jpeg"))]
			for file in img_files:
				path = os.path.join(person_dir, file)
				image = cv2.imread(path)
				if image is None:
					continue
				image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

				try:
					boxes, _ = detector.detect(image_rgb)
				except Exception as e:
					print(f"[WARN] Detection failed for {path}: {e}")
					continue
				if boxes is None or len(boxes) == 0:
					continue

				try:
					aligned = detector.extract(image_rgb, boxes, save_path=None)  # (N,3,160,160) float [0,1]
				except Exception as e:
					print(f"[WARN] Alignment failed for {path}: {e}")
					continue
				if aligned is None or aligned.shape[0] == 0:
					continue

				with torch.no_grad():
					aligned = fixed_image_standardization(aligned.to(device))
					embs = embedder(aligned).detach().cpu().numpy()

				for e in embs:
					encodings.append(np.asarray(e, dtype=np.float32))
					names.append(person_name)

	print(f"[INFO] Loaded {len(encodings)} face encodings from {folder}.")
	return encodings, names
