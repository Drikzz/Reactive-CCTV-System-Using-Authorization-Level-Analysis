import cv2
import numpy as np
from keras_facenet import FaceNet

_embedder = None

def get_embedder():
	"""Lazy-load and return a singleton FaceNet embedder."""
	global _embedder
	if _embedder is None:
		_embedder = FaceNet()
	return _embedder

def compute_embedding_distance(embedding, centroid):
	"""Compute Euclidean distance between embedding and class centroid."""
	return np.linalg.norm(embedding - centroid)

def recognize_faces(frame, classifier=None, label_encoder=None, threshold=0.5, centroids=None, dist_threshold=1.0):
	"""Detect faces, embed with FaceNet, classify with SVM."""
	rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
	try:
		detections = get_embedder().extract(rgb, threshold=0.95)
	except Exception as e:
		print(f"[ERROR] FaceNet extraction failed: {e}")
		return []

	recognized = []
	for det in detections:
		box = det.get('box')      # (x, y, w, h)
		embedding = det.get('embedding')
		if embedding is None or box is None:
			continue

		x, y, w, h = box
		x1, y1 = max(0, x), max(0, y)
		x2, y2 = x1 + max(0, w), y1 + max(0, h)
		# ensure integer bbox
		x1, y1, x2, y2 = map(int, (x1, y1, x2, y2))

		name, prob = "Unknown", 0.0
		distance = None
		
		try:
			emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
			if classifier is not None and label_encoder is not None:
				probs = classifier.predict_proba(emb)[0]
				pred_idx = int(np.argmax(probs))
				max_prob = float(probs[pred_idx])
				
				# Get predicted class
				encoded_label = classifier.classes_[pred_idx]
				pred_name = label_encoder.inverse_transform([encoded_label])[0]
				
				# Open-set recognition using both probability and distance
				is_known = max_prob >= threshold
				
				# Check embedding distance if centroids are available
				if centroids is not None and pred_name in centroids:
					distance = compute_embedding_distance(embedding, centroids[pred_name])
					is_known = is_known and distance <= dist_threshold
					print(f"[DEBUG] Face prediction: {pred_name}, prob={max_prob:.3f}, distance={distance:.3f}, threshold={dist_threshold:.3f}, is_known={is_known}")
				else:
					print(f"[DEBUG] Face prediction: {pred_name}, prob={max_prob:.3f}, is_known={is_known}")
				
				if is_known:
					name = pred_name
					prob = max_prob
		except Exception as e:
			print(f"[WARN] Classification failed: {e}")

		recognized.append({"name": name, "bbox": (x1, y1, x2, y2), "prob": prob, "distance": distance})
	return recognized
