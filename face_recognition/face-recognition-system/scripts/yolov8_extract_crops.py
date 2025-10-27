import argparse
import os
from pathlib import Path
from datetime import datetime

# add default video path (can be overridden by env var VIDEO_PATH or --source)
VIDEO_PATH = r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\Mp4TESTING\ThesisMP4TEST5.mp4"
VIDEO_PATH = os.getenv("VIDEO_PATH", VIDEO_PATH)

def main():
    parser = argparse.ArgumentParser(description="Run YOLOv8 face detector on a video and save face crops")
    # make --source optional; allow --video alias; fallback to env VIDEO_PATH / default
    parser.add_argument("--source", help="Path to input video file (overrides VIDEO_PATH env/default)", default=None)
    parser.add_argument("--video", help=argparse.SUPPRESS, default=None)  # hidden alias if you prefer --video
    parser.add_argument(
        "--output",
        default=r"C:\Users\Alexa\OneDrive\Documents\Thesis\Reactive-CCTV-System-Using-Authorization-Level-Analysis\face_recognition\face-recognition-system\data\face_crops",
        help="Directory to save face crops"
    )
    # default to ultralytics small model (will be downloaded if not present)
    parser.add_argument("--model", default="yolov8n.pt",
                        help="YOLOv8 model path or name (e.g. 'yolov8n.pt' or path/to/model.pt)")
    parser.add_argument("--conf", type=float, default=0.35, help="Detection confidence threshold")
    parser.add_argument("--min-size", type=int, default=30, help="Minimum face box width/height to save")
    parser.add_argument("--max-frames", type=int, default=0, help="Max frames to process (0 = all)")
    parser.add_argument("--frame-skip", type=int, default=1, help="Process every Nth frame (default 1 = every frame)")
    parser.add_argument("--max-per-frame", type=int, default=1, help="Max detections to save per frame (highest-confidence first)")
    parser.add_argument("--min-area-frac", type=float, default=0.001, help="Minimum box area fraction of frame (w*h / frame_area) to keep")
    parser.add_argument("--nms-iou", type=float, default=0.3, help="IoU threshold for per-frame NMS (lower -> fewer overlapping crops)")
    parser.add_argument("--min-center-dist", type=float, default=0.0, help="Minimum center distance (pixels) to treat boxes as distinct; 0 to disable")
    parser.add_argument("--track", action="store_true", help="Run tracker (model.track) over the video and save tracked face crops")
    parser.add_argument("--tracker", type=str, default="bytetrack.yaml", help="Tracker config name or path (e.g. 'bytetrack.yaml')")
    parser.add_argument("--track-class", type=str, default="face", help="Class name to track (when model exposes names); use empty to accept all")
    parser.add_argument("--verify-haar", action="store_true", help="Verify detected crop contains a face using Haar cascade (reduces false positives)")
    parser.add_argument("--require-face-class", action="store_true",
                        help="Require model to have a 'face' class and only save detections of that class")
    parser.add_argument("--debug", action="store_true", help="Print debug info")
    args = parser.parse_args()

    # select source: CLI --source > --video alias > env VIDEO_PATH > default VIDEO_PATH
    src_arg = args.source or args.video or os.getenv("VIDEO_PATH") or VIDEO_PATH
    source = Path(src_arg)
    out_dir = Path(args.output)
    model_path = str(Path(args.model))

    if not source.exists():
        print(f"ERROR: source video not found: {source}")
        return

    out_dir.mkdir(parents=True, exist_ok=True)

    try:
        from ultralytics import YOLO
    except Exception:
        print("ERROR: ultralytics not installed. Install with: pip install ultralytics")
        return

    # try to load requested model, fallback to yolov8n.pt if loading fails
    try:
        model = YOLO(model_path)
    except Exception as e:
        print(f"[WARN] Failed to load model '{model_path}': {e}")
        if model_path != "yolov8n.pt":
            try:
                print("[INFO] Falling back to 'yolov8n.pt' (will download if needed).")
                model = YOLO("yolov8n.pt")
            except Exception as e2:
                print(f"ERROR: cannot load fallback model: {e2}")
                return
        else:
            return

    # attempt to detect face class index(es)
    allowed_class_ids = []
    names = None
    try:
        # ultralytics model may expose .model.names or .names
        if hasattr(model, "model") and hasattr(model.model, "names"):
            names = model.model.names
        elif hasattr(model, "names"):
            names = model.names

        if names:
            # prefer explicit 'face' classes; also accept 'head'
            allowed_class_ids = [int(k) for k, v in names.items() if any(t in str(v).lower() for t in ("face", "head"))]
            # explicitly exclude generic 'person','human','body' classes if present
            excluded = {int(k) for k, v in names.items() if any(t in str(v).lower() for t in ("person", "human", "body"))}
            allowed_class_ids = [cid for cid in allowed_class_ids if cid not in excluded]

            if allowed_class_ids:
                print(f"[INFO] Model face/head class id(s): {allowed_class_ids} -> {[names[i] for i in allowed_class_ids]}")
            else:
                if args.debug:
                    print(f"[INFO] Model class names: {names}")
                print("[INFO] Model has no explicit 'face'/'head' class; will rely on Haar/face verification to filter crops.")
    except Exception:
        if args.debug:
            print("[DEBUG] Failed to read model class names")

    if args.require_face_class and not allowed_class_ids:
        print("ERROR: --require-face-class specified but model has no 'face' class. Aborting.")
        return

    import cv2
    vidcap = cv2.VideoCapture(str(source))
    if not vidcap.isOpened():
        print("ERROR: cannot open video:", source)
        return

    frame_idx = 0
    saved = 0
    video_stem = source.stem

    def _to_numpy(x):
        try:
            import numpy as _np
        except Exception:
            return x
        if x is None:
            return None
        try:
            import torch
            if isinstance(x, torch.Tensor):
                return x.detach().cpu().numpy()
        except Exception:
            pass
        try:
            return _np.asarray(x)
        except Exception:
            return x

    def _iou(a, b):
        # a,b = [x1,y1,x2,y2]
        xa1, ya1, xa2, ya2 = a
        xb1, yb1, xb2, yb2 = b
        xi1 = max(xa1, xb1); yi1 = max(ya1, yb1)
        xi2 = min(xa2, xb2); yi2 = min(ya2, yb2)
        iw = max(0, xi2 - xi1); ih = max(0, yi2 - yi1)
        inter = iw * ih
        area_a = max(0, (xa2 - xa1)) * max(0, (ya2 - ya1))
        area_b = max(0, (xb2 - xb1)) * max(0, (yb2 - yb1))
        union = area_a + area_b - inter
        return inter / union if union > 0 else 0.0

    def non_max_suppression(boxes_xyxy, scores, iou_thresh=0.4):
        # boxes_xyxy: Nx4 numpy array, scores: N
        import numpy as _np
        if boxes_xyxy is None or len(boxes_xyxy) == 0:
            return _np.array([], dtype=int)
        idxs = _np.argsort(-scores)
        keep = []
        for i in idxs:
            keep_flag = True
            for j in keep:
                if _iou(boxes_xyxy[i], boxes_xyxy[j]) > iou_thresh:
                    keep_flag = False
                    break
            if keep_flag:
                keep.append(i)
        return _np.array(keep, dtype=int)

    def verify_with_haar(crop_bgr, min_size):
        """
        Return True if a frontal face is found inside crop_bgr using OpenCV Haar cascade.
        Fast and effective to filter non-face detections from generic COCO models.
        """
        try:
            import cv2
        except Exception:
            return False
        if crop_bgr is None:
            return False
        try:
            gray = cv2.cvtColor(crop_bgr, cv2.COLOR_BGR2GRAY)
            cascade_path = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
            face_cascade = cv2.CascadeClassifier(cascade_path)
            faces = face_cascade.detectMultiScale(
                gray,
                scaleFactor=1.05,
                minNeighbors=4,
                minSize=(max(20, min_size // 3), max(20, min_size // 3))
            )
            return len(faces) > 0
        except Exception:
            return False

    while True:
        ret, frame = vidcap.read()
        if not ret:
            break
        frame_idx += 1
        if args.max_frames and frame_idx > args.max_frames:
            break
        # skip frames to reduce number of crops / work
        if args.frame_skip and args.frame_skip > 1 and (frame_idx - 1) % args.frame_skip != 0:
            continue

        # run YOLO on this frame (use model.predict with stream=False for accurate boxes)
        try:
            results = model.predict(source=[frame], conf=args.conf, imgsz=640, verbose=False)
        except Exception:
            try:
                # fallback call signature
                results = model(frame, conf=args.conf, imgsz=640)
            except Exception as e:
                if args.debug:
                    print(f"[DEBUG] prediction failed on frame {frame_idx}: {e}")
                continue

        if not results:
            continue
        res = results[0]
        boxes = getattr(res, "boxes", None)
        if boxes is None:
            continue

        # extract coords, scores and classes (if available) safely
        xyxy_attr = getattr(boxes, "xyxy", None)
        xyxy = _to_numpy(xyxy_attr) if xyxy_attr is not None else None

        conf_attr = getattr(boxes, "conf", None)
        confs = _to_numpy(conf_attr) if conf_attr is not None else None

        cls_arr = None
        try:
            cls_attr = getattr(boxes, "cls", None)
            if cls_attr is None:
                cls_attr = getattr(boxes, "classes", None)
            cls_arr = _to_numpy(cls_attr) if cls_attr is not None else None
        except Exception:
            cls_arr = None

        if xyxy is None:
            continue

        # perform per-frame NMS to remove overlapping/duplicate boxes
        try:
            import numpy as _np
            boxes_np = _np.asarray(xyxy)  # shape (N,4)
            confs_arr = _np.asarray(confs) if confs is not None else _np.zeros(len(boxes_np))
            keep_idxs = non_max_suppression(boxes_np, confs_arr, iou_thresh=float(args.nms_iou))
            if args.min_center_dist and args.min_center_dist > 0 and len(keep_idxs) > 1:
                # further remove near-duplicate boxes by center distance
                centers = [( (boxes_np[i][0]+boxes_np[i][2])/2.0, (boxes_np[i][1]+boxes_np[i][3])/2.0 ) for i in keep_idxs]
                filtered = []
                for ii, i in enumerate(keep_idxs):
                    cx, cy = centers[ii]
                    too_close = False
                    for jj, j in enumerate(filtered):
                        cx2, cy2 = centers[jj]
                        if ((cx - cx2)**2 + (cy - cy2)**2) ** 0.5 < float(args.min_center_dist):
                            too_close = True
                            break
                    if not too_close:
                        filtered.append(ii)
                # filtered contains indices into keep_idxs; convert to actual indices
                keep_idxs = keep_idxs[_np.asarray(filtered, dtype=int)]
            # order kept by descending confidence
            order = keep_idxs[_np.argsort(-confs_arr[keep_idxs])] if len(keep_idxs) > 0 else []
        except Exception:
            order = list(range(len(xyxy)))

        kept = 0
        frame_area = float(frame.shape[0] * frame.shape[1])
        for rank_idx in order:
            if kept >= args.max_per_frame:
                break
            try:
                b = xyxy[rank_idx]
                c = float(confs[rank_idx]) if (confs is not None and rank_idx < len(confs)) else 0.0
                class_id = int(cls_arr[rank_idx]) if (cls_arr is not None and rank_idx < len(cls_arr)) else None
            except Exception:
                continue

            # enforce face/head-class if model provides such classes
            if allowed_class_ids and (class_id is None or class_id not in allowed_class_ids):
                # skip non-face/body classes
                continue

            x1, y1, x2, y2 = map(int, b)
            # clamp boxes
            H, W = frame.shape[:2]
            x1, y1 = max(0, x1), max(0, y1)
            x2, y2 = min(W, x2), min(H, y2)
            w, h = x2 - x1, y2 - y1
            if w < args.min_size or h < args.min_size:
                continue

            # require minimum proportion of frame area
            if (w * h) / frame_area < float(args.min_area_frac):
                continue

            crop = frame[y1:y2, x1:x2]
            # if model has no explicit face/head class, or user requested, verify crop contains a face
            auto_verify = args.verify_haar or (not allowed_class_ids)
            if auto_verify:
                try:
                    if not verify_with_haar(crop, args.min_size):
                        if args.debug:
                            print(f"[DEBUG] Haar verification failed for frame {frame_idx} crop {rank_idx}, skipping")
                        continue
                except Exception as e:
                    if args.debug:
                        print("[DEBUG] Haar verification exception:", e)
                    continue

            ts = datetime.now().strftime("%Y%m%d_%H%M%S")
            fname = f"{video_stem}_f{frame_idx:06d}_{rank_idx}_c{int(c*100):02d}_{ts}.jpg"
            out_path = out_dir / fname
            try:
                cv2.imwrite(str(out_path), crop)
                saved += 1
                kept += 1
            except Exception:
                pass

        # optional small progress print
        if frame_idx % 100 == 0:
            print(f"Processed {frame_idx} frames, saved {saved} crops...")

    # if tracking requested, run tracking pipeline (ultralytics model.track) and save crops per tracked box
    if args.track:
        print("[INFO] Running tracker:", args.tracker)
        try:
            results = model.track(source=str(source), tracker=args.tracker, conf=args.conf, imgsz=640, persist=True)
        except Exception as e:
            print("ERROR: model.track failed:", e)
            return

        # determine face class ids if provided
        class_names = None
        try:
            if hasattr(model, "model") and hasattr(model.model, "names"):
                class_names = model.model.names
            elif hasattr(model, "names"):
                class_names = model.names
        except Exception:
            class_names = None

        for frame_idx, res in enumerate(results, start=1):
            boxes = getattr(res, "boxes", None)
            if boxes is None:
                continue
            xyxy = _to_numpy(getattr(boxes, "xyxy", None))
            confs = _to_numpy(getattr(boxes, "conf", None))
            cls_arr = None
            try:
                cls_attr = getattr(boxes, "cls", None)
                if cls_attr is None:
                    cls_attr = getattr(boxes, "classes", None)
                cls_arr = _to_numpy(cls_attr) if cls_attr is not None else None
            except Exception:
                cls_arr = None
            # tracked ids (per-box)
            ids = None
            try:
                ids = _to_numpy(getattr(boxes, "id", None) or getattr(boxes, "ids", None) or getattr(boxes, "tracker_id", None))
            except Exception:
                ids = None

            if xyxy is None:
                continue

            # save each tracked detection (optionally filter by class name)
            for i in range(len(xyxy)):
                try:
                    b = xyxy[i]
                    c = float(confs[i]) if (confs is not None and i < len(confs)) else 0.0
                    class_id = int(cls_arr[i]) if (cls_arr is not None and i < len(cls_arr)) else None
                    track_id = int(ids[i]) if (ids is not None and i < len(ids)) else None
                except Exception:
                    continue

                # if model exposes names and user requested a specific track-class, enforce it
                if args.track_class and class_names:
                    name = str(class_names.get(class_id, "")).lower() if class_id is not None else ""
                    if args.track_class.lower() not in name:
                        continue

                x1, y1, x2, y2 = map(int, b)
                H, W = res.orig_shape[0], res.orig_shape[1] if hasattr(res, "orig_shape") else (None, None)
                # fallback to frame dimensions if res doesn't expose orig_shape
                if W is None or H is None:
                    import cv2
                    cap = cv2.VideoCapture(str(source))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx-1))
                    ret, tmpf = cap.read()
                    cap.release()
                    if ret:
                        H, W = tmpf.shape[:2]
                if W is None or H is None:
                    H, W = y2 - y1, x2 - x1

                x1, y1 = max(0, x1), max(0, y1)
                x2, y2 = min(W, x2), min(H, y2)
                if x2 <= x1 or y2 <= y1:
                    continue

                crop = res.orig_img[int(y1):int(y2), int(x1):int(x2)] if hasattr(res, "orig_img") else None
                # fallback: read frame and crop
                if crop is None:
                    import cv2
                    cap = cv2.VideoCapture(str(source))
                    cap.set(cv2.CAP_PROP_POS_FRAMES, max(0, frame_idx-1))
                    ret, frame = cap.read()
                    cap.release()
                    if not ret:
                        continue
                    crop = frame[y1:y2, x1:x2]

                ts = datetime.now().strftime("%Y%m%d_%H%M%S")
                tid_str = f"_id{track_id}" if track_id is not None else ""
                fname = f"{video_stem}_trk{tid_str}_f{frame_idx:06d}_{i}_c{int(c*100):02d}_{ts}.jpg"
                out_path = out_dir / fname
                try:
                    import cv2
                    cv2.imwrite(str(out_path), crop)
                except Exception:
                    pass

        print("Done tracking; saved tracked crops to", out_dir)
        return

    vidcap.release()
    print(f"Done. Processed {frame_idx} frames, saved {saved} face crops to {out_dir}")

if __name__ == "__main__":
    main()