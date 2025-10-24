import cv2
import numpy as np

class MTCNNWrapper:
    """
    Unified MTCNN wrapper:
    - preferrs facenet-pytorch MTCNN (supports image_size)
    - falls back to ipazc mtcnn
    - provides extract_crops(frame) -> list of (crop_bgr, box, score)
    """
    def __init__(self, image_size=160, margin=0, keep_all=True, device=None):
        self.image_size = int(image_size)
        self.margin = int(margin)
        self.keep_all = keep_all
        self.device = device
        self.backend = None
        self.mtcnn = None

        # try facenet-pytorch first
        try:
            from facenet_pytorch import MTCNN as MTCNN_FP
            # facenet-pytorch accepts image_size
            self.mtcnn = MTCNN_FP(image_size=self.image_size, margin=self.margin, keep_all=self.keep_all, device=self.device)
            self.backend = "facenet-pytorch"
        except Exception:
            # fallback to ipazc/mtcnn
            try:
                from mtcnn import MTCNN as MTCNN_IP
                # ipazc mtcnn constructor doesn't accept image_size
                self.mtcnn = MTCNN_IP()
                self.backend = "ipazc-mtcnn"
            except Exception as e:
                raise RuntimeError("No MTCNN implementation available (install facenet-pytorch or mtcnn). Error: " + str(e))

    def _detect_boxes(self, frame_rgb):
        """
        Return list of (x1,y1,x2,y2, score)
        """
        # facenet-pytorch: mtcnn.detect
        if self.backend == "facenet-pytorch":
            try:
                boxes, probs = self.mtcnn.detect(frame_rgb)
                if boxes is None:
                    return []
                out = []
                for b, p in zip(boxes, probs):
                    x1, y1, x2, y2 = map(int, b)
                    out.append((x1, y1, x2, y2, float(p) if p is not None else 0.0))
                return out
            except TypeError:
                # defensive: some versions behave unexpectedly — fallback below
                pass

        # ipazc mtcnn: try detect_faces, else detect
        try:
            faces = self.mtcnn.detect_faces(frame_rgb)
            # detect_faces returns list of dicts with 'box' [x,y,w,h] and 'confidence'
            out = []
            for f in faces:
                box = f.get("box") or f.get("bbox")
                conf = f.get("confidence") or f.get("probability") or 0.0
                if box:
                    x, y, w, h = map(int, box)
                    out.append((x, y, x + w, y + h, float(conf)))
            return out
        except Exception:
            pass

        # try generic detect (some builds expose detect)
        try:
            boxes, probs = self.mtcnn.detect(frame_rgb)
            if boxes is None:
                return []
            out = []
            for b, p in zip(boxes, probs):
                x1, y1, x2, y2 = map(int, b)
                out.append((x1, y1, x2, y2, float(p) if p is not None else 0.0))
            return out
        except Exception:
            return []

    def extract_crops(self, frame_bgr):
        """
        Input: OpenCV BGR frame (numpy array)
        Output: list of tuples (crop_bgr_resized, (x1,y1,x2,y2), score)
        """
        if frame_bgr is None:
            return []

        frame_rgb = cv2.cvtColor(frame_bgr, cv2.COLOR_BGR2RGB)
        boxes = self._detect_boxes(frame_rgb)
        crops = []
        H, W = frame_bgr.shape[:2]

        for (x1, y1, x2, y2, score) in boxes:
            # apply margin (in pixels)
            w = x2 - x1
            h = y2 - y1
            mx = int(self.margin * w / max(1, self.image_size)) if self.margin else 0
            my = int(self.margin * h / max(1, self.image_size)) if self.margin else 0
            x1m = max(0, x1 - mx)
            y1m = max(0, y1 - my)
            x2m = min(W, x2 + mx)
            y2m = min(H, y2 + my)

            if x2m <= x1m or y2m <= y1m:
                continue

            crop = frame_bgr[y1m:y2m, x1m:x2m]
            try:
                crop_resized = cv2.resize(crop, (self.image_size, self.image_size), interpolation=cv2.INTER_AREA)
            except Exception:
                # fallback: return original crop if resize failed
                crop_resized = crop
            crops.append((crop_resized, (x1m, y1m, x2m, y2m), float(score)))
            # if not keep_all, return first
            if not self.keep_all:
                break

        return crops