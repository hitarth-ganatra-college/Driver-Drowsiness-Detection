import os
import cv2
import numpy as np
import logging

# Standard landmark indices (consistent across all MediaPipe 468/478 FaceMesh models)
LEFT_EYE = [362, 385, 387, 263, 373, 380]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]
MOUTH = [61, 291, 13, 14, 78, 308]
ALL_EYE_IDXS = set(LEFT_EYE + RIGHT_EYE)

_task_path = os.path.join(os.path.dirname(__file__), "face_landmarker.task")
_landmarker = None

def get_landmarker():
    global _landmarker
    if _landmarker is not None:
        return _landmarker
    try:
        import mediapipe as mp
        from mediapipe.tasks import python
        from mediapipe.tasks.python import vision
        if os.path.exists(_task_path):
            base_options = python.BaseOptions(model_asset_path=_task_path)
            options = vision.FaceLandmarkerOptions(
                base_options=base_options,
                num_faces=1,
                min_face_detection_confidence=0.5,
                min_face_presence_confidence=0.5
            )
            _landmarker = vision.FaceLandmarker.create_from_options(options)
            logging.info("Initialized MediaPipe FaceLandmarker with XNNPACK")
    except Exception as e:
        logging.warning(f"Could not load MediaPipe Tasks FaceLandmarker: {e}")
    return _landmarker

class NormalizedLandmark:
    def __init__(self, x, y, z=0.0):
        self.x = float(x)
        self.y = float(y)
        self.z = float(z)

def detect_landmarks(image_bgr):
    """
    Detects face landmarks on an image.
    Returns: list of NormalizedLandmark objects, or None if no face detected.
    """
    if image_bgr is None:
        return None

    h, w = image_bgr.shape[:2]
    landmarker = get_landmarker()

    if landmarker is not None:
        try:
            import mediapipe as mp
            image_rgb = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB)
            mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=image_rgb)
            result = landmarker.detect(mp_image)
            if result.face_landmarks and len(result.face_landmarks) > 0:
                raw_landmarks = result.face_landmarks[0]
                return [NormalizedLandmark(lm.x, lm.y, lm.z) for lm in raw_landmarks]
        except Exception as e:
            logging.debug(f"MediaPipe Tasks inference failed: {e}")

    # Fallback to legacy mp.solutions if present
    try:
        import mediapipe as mp
        if hasattr(mp, 'solutions') and hasattr(mp.solutions, 'face_mesh'):
            with mp.solutions.face_mesh.FaceMesh(
                static_image_mode=True,
                max_num_faces=1,
                min_detection_confidence=0.5
            ) as face_mesh:
                results = face_mesh.process(cv2.cvtColor(image_bgr, cv2.COLOR_BGR2RGB))
                if results.multi_face_landmarks:
                    return [NormalizedLandmark(lm.x, lm.y, lm.z) for lm in results.multi_face_landmarks[0].landmark]
    except Exception:
        pass

    return None

def draw_face_landmarks(img, landmarks, circle_color=(255, 255, 255), circle_radius=2):
    """Draws key landmarks onto img in-place."""
    h, w = img.shape[:2]
    for lm in landmarks:
        cx, cy = int(lm.x * w), int(lm.y * h)
        if 0 <= cx < w and 0 <= cy < h:
            cv2.circle(img, (cx, cy), circle_radius, circle_color, -1)
    return img
