import cv2
import numpy as np
import torch
from facenet_pytorch import MTCNN, InceptionResnetV1, fixed_image_standardization

_device = 'cuda' if torch.cuda.is_available() else 'cpu'
_embedder = None
_detector = None

def get_embedder():
	"""Lazy-load and return a singleton InceptionResnetV1 embedder."""
	global _embedder
	if _embedder is None:
		_embedder = InceptionResnetV1(pretrained='vggface2').to(_device).eval()
	return _embedder

def get_detector():
	"""Lazy-load and return a singleton MTCNN detector."""
	global _detector
	if _detector is None:
		_detector = MTCNN(image_size=160, margin=0, keep_all=True, device=_device, post_process=False)
	return _detector

def compute_embedding_distance(embedding, centroid):
	"""Compute Euclidean distance between embedding and class centroid."""
	return float(np.linalg.norm(np.asarray(embedding, dtype=np.float32) - np.asarray(centroid, dtype=np.float32)))

def recognize_faces(frame, classifier=None, label_encoder=None, threshold=0.5, centroids=None, dist_threshold=1.0):
	"""
	Detect, embed, and classify faces in a frame.
	Returns: list of dicts: {name, bbox(x1,y1,x2,y2), prob, distance}
	Optimized for live camera use.
	"""
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	det = get_detector()
	emb_model = get_embedder()

	try:
		boxes, _ = det.detect(rgb)
		if boxes is None or len(boxes) == 0:
			return []
		aligned = det.extract(rgb, boxes, save_path=None)
		if aligned is None or aligned.shape[0] == 0:
			return []
	except Exception as e:
		print(f"[ERROR] MTCNN detection/extract failed: {e}")
		return []

	with torch.no_grad():
		aligned_std = fixed_image_standardization(aligned.to(_device))
		embeddings = emb_model(aligned_std).cpu().numpy()  # (N,512)

	results = []
	for i, box in enumerate(boxes):
		x1, y1, x2, y2 = map(lambda v: max(0, int(v)), box)
		name, prob, distance = "Unknown", 0.0, None

		if classifier and label_encoder:
			feat = embeddings[i].reshape(1, -1)
			try:
				probs_svm = classifier.predict_proba(feat)[0]
				idx = int(np.argmax(probs_svm))
				max_prob = float(probs_svm[idx])
				pred_name = label_encoder.inverse_transform([classifier.classes_[idx]])[0]

				distance = None
				if centroids and pred_name in centroids:
					distance = compute_embedding_distance(embeddings[i], centroids[pred_name])

				# Final decision
				if max_prob >= threshold and (distance is None or distance <= dist_threshold):
					name, prob = pred_name, max_prob
			except Exception as e:
				print(f"[WARN] Classification failed for face {i}: {e}")

		results.append({"name": name, "bbox": (x1, y1, x2, y2), "prob": prob, "distance": distance})

	return results