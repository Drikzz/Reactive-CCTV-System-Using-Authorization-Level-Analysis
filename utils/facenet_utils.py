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

def recognize_faces(frame, classifier=None, label_encoder=None, threshold=0.5):
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
		try:
			emb = np.asarray(embedding, dtype=np.float32).reshape(1, -1)
			if classifier is not None and label_encoder is not None:
				probs = classifier.predict_proba(emb)[0]
				pred_idx = int(np.argmax(probs))
				max_prob = float(probs[pred_idx])
				if max_prob >= threshold:
					# map probability index -> encoded label -> original name
					encoded_label = classifier.classes_[pred_idx]
					name = label_encoder.inverse_transform([encoded_label])[0]
					prob = max_prob
		except Exception as e:
			print(f"[WARN] Classification failed: {e}")

		recognized.append({"name": name, "bbox": (x1, y1, x2, y2), "prob": prob})
	return recognized
