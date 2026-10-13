import os
os.environ.setdefault('KERAS_BACKEND', 'torch')
from model import load_saved_model, predict, predict_fatigue_score
import numpy as np
import cv2
import tempfile
import logging
import mediapipe as mp

try:
    model = load_saved_model(load_last=True)
except Exception as e:
    logging.warning(f"Could not initialize deep learning model: {e}")
    model = None

from face_detector_adapter import detect_landmarks, draw_face_landmarks, LEFT_EYE, RIGHT_EYE, MOUTH

# State tracking for duration thresholding
eye_closed_counter = 0
yawn_counter = 0

# Baseline Thresholds & Settings
EYE_CLOSED_THRESHOLD = 0.22
YAWN_THRESHOLD = 0.60
FRAMES_EYE_CLOSED = 15
FRAMES_YAWNING = 15

# Calibration variables
CALIBRATION_TOTAL_FRAMES = 30
calibration_frames = 0
calibration_ear_samples = []
is_calibrated = False

# CNN inference decimation (smoother real-time performance)
frame_counter = 0
last_cnn_fatigue_prob = 0.0

def reset_calibration():
    global calibration_frames, calibration_ear_samples, is_calibrated, EYE_CLOSED_THRESHOLD
    calibration_frames = 0
    calibration_ear_samples = []
    is_calibrated = False
    EYE_CLOSED_THRESHOLD = 0.22

def calibrate_step(ear_value):
    global calibration_frames, calibration_ear_samples, is_calibrated, EYE_CLOSED_THRESHOLD
    if calibration_frames < CALIBRATION_TOTAL_FRAMES:
        calibration_ear_samples.append(ear_value)
        calibration_frames += 1
        if calibration_frames >= CALIBRATION_TOTAL_FRAMES:
            baseline_ear = float(np.mean(calibration_ear_samples))
            EYE_CLOSED_THRESHOLD = float(np.clip(baseline_ear * 0.75, 0.17, 0.28))
            is_calibrated = True

def calculate_distance(p1, p2):
    return np.linalg.norm(np.array(p1) - np.array(p2))

def calculate_ear(landmarks, eye_indices, img_w, img_h):
    coords = [(int(landmarks[idx].x * img_w), int(landmarks[idx].y * img_h)) for idx in eye_indices]
    p1, p2, p3, p4, p5, p6 = coords
    # EAR = (||p2 - p6|| + ||p3 - p5||) / (2 * ||p1 - p4||)
    vertical_1 = calculate_distance(p2, p6)
    vertical_2 = calculate_distance(p3, p5)
    horizontal = calculate_distance(p1, p4)
    if horizontal == 0:
        return 0.0
    return (vertical_1 + vertical_2) / (2.0 * horizontal)

def calculate_mar(landmarks, mouth_indices, img_w, img_h):
    c1 = (int(landmarks[mouth_indices[0]].x * img_w), int(landmarks[mouth_indices[0]].y * img_h))
    c2 = (int(landmarks[mouth_indices[1]].x * img_w), int(landmarks[mouth_indices[1]].y * img_h))
    upper = (int(landmarks[mouth_indices[2]].x * img_w), int(landmarks[mouth_indices[2]].y * img_h))
    lower = (int(landmarks[mouth_indices[3]].x * img_w), int(landmarks[mouth_indices[3]].y * img_h))
    
    horizontal = calculate_distance(c1, c2)
    vertical = calculate_distance(upper, lower)
    if horizontal == 0:
        return 0.0
    return vertical / horizontal

def process_frame_drowsiness(img, calibrate_mode=False):
    global eye_closed_counter, yawn_counter, frame_counter, last_cnn_fatigue_prob
    
    h, w = img.shape[:2]
    face_landmarks = detect_landmarks(img)
    
    status = "Active"
    is_drowsy = False
    cnn_score = last_cnn_fatigue_prob
    fusion_score = 0.0
    
    if face_landmarks:
        
        left_ear = calculate_ear(face_landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = calculate_mar(face_landmarks, MOUTH, w, h)
        
        # Handle calibration if enabled
        if calibrate_mode and not is_calibrated:
            calibrate_step(avg_ear)
            
        # Check eye closure
        if avg_ear < EYE_CLOSED_THRESHOLD:
            eye_closed_counter += 1
        else:
            eye_closed_counter = max(0, eye_closed_counter - 1)
            
        # Check yawning
        if mar > YAWN_THRESHOLD:
            yawn_counter += 1
        else:
            yawn_counter = max(0, yawn_counter - 1)
            
        # Draw eye points
        for idx in LEFT_EYE + RIGHT_EYE:
            cx, cy = int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)
            cv2.circle(img, (cx, cy), 2, (0, 255, 0), -1)
            
        # Draw mouth points
        for idx in [13, 14, 61, 291]:
            cx, cy = int(face_landmarks[idx].x * w), int(face_landmarks[idx].y * h)
            cv2.circle(img, (cx, cy), 2, (255, 255, 0), -1)

        # Dual-stream: Run CNN inference every 3 frames for optimal real-time performance
        frame_counter += 1
        if model is not None and frame_counter % 3 == 0:
            try:
                xs = [int(lm.x * w) for lm in face_landmarks]
                ys = [int(lm.y * h) for lm in face_landmarks]
                xmin, xmax = max(0, min(xs)), min(w, max(xs))
                ymin, ymax = max(0, min(ys)), min(h, max(ys))
                face_crop = img[ymin:ymax, xmin:xmax]
                if face_crop.size > 0:
                    prob, _ = predict_fatigue_score(model, face_crop)
                    last_cnn_fatigue_prob = prob
                    cnn_score = prob
            except Exception as e:
                logging.debug(f"CNN execution skipped: {e}")

        # Geometric fatigue score [0.0 - 1.0]
        geo_eye_score = min(1.0, eye_closed_counter / FRAMES_EYE_CLOSED)
        geo_yawn_score = min(1.0, yawn_counter / FRAMES_YAWNING)
        geo_score = max(geo_eye_score, geo_yawn_score)

        # Hybrid AI Fusion Score: combines Geometric rules (60%) and CNN Deep Learning (40%)
        if model is not None:
            fusion_score = 0.6 * geo_score + 0.4 * cnn_score
            is_drowsy = (fusion_score >= 0.55) or (eye_closed_counter >= FRAMES_EYE_CLOSED) or (yawn_counter >= FRAMES_YAWNING)
        else:
            fusion_score = geo_score
            is_drowsy = (eye_closed_counter >= FRAMES_EYE_CLOSED) or (yawn_counter >= FRAMES_YAWNING)

        # Trigger alerts
        if eye_closed_counter >= FRAMES_EYE_CLOSED:
            status = "DROWSY! (Eyes Closed)"
        elif yawn_counter >= FRAMES_YAWNING:
            status = "DROWSY! (Yawning)"
        elif is_drowsy:
            status = "DROWSY! (High Fatigue)"

        # Telemetry HUD overlay
        ear_text = f"EAR: {avg_ear:.2f} (Thresh: {EYE_CLOSED_THRESHOLD:.2f})"
        mar_text = f"MAR: {mar:.2f} (Thresh: {YAWN_THRESHOLD:.2f})"
        cnn_text = f"CNN Conf: {cnn_score * 100:.1f}%" if model is not None else "CNN: Standby"
        fusion_text = f"Fatigue Index: {fusion_score * 100:.1f}%"

        cv2.putText(img, ear_text, (20, 35), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, mar_text, (20, 65), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (255, 255, 255), 2)
        cv2.putText(img, cnn_text, (20, 95), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (200, 255, 200), 2)
        cv2.putText(img, fusion_text, (20, 125), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (180, 220, 255), 2)

        if calibrate_mode and not is_calibrated:
            calib_pct = int((calibration_frames / CALIBRATION_TOTAL_FRAMES) * 100)
            cv2.putText(img, f"Calibrating: {calib_pct}%", (20, 155), cv2.FONT_HERSHEY_SIMPLEX, 0.65, (0, 255, 255), 2)
    else:
        # No face detected
        cv2.putText(img, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        eye_closed_counter = 0
        yawn_counter = 0

    # Draw Status banner
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    banner_y = 155 if (not (calibrate_mode and not is_calibrated)) else 185
    cv2.putText(img, f"Status: {status}", (20, banner_y), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
    if is_drowsy:
        overlay = img.copy()
        cv2.rectangle(overlay, (0, 0), (w, h), (0, 0, 255), -1)
        cv2.addWeighted(overlay, 0.35, img, 0.65, 0, img)
        
    return img, is_drowsy

def predict_rt(img):
    img_processed, is_drowsy = process_frame_drowsiness(img)
    return is_drowsy

def predict_video(video_file):
    video_name = video_file.name 
    tempVidFile = tempfile.NamedTemporaryFile(delete=False, suffix='.mp4')
    tempVidFile.write(video_file.read())
    tempVidFile.close()

    vidCap = cv2.VideoCapture(tempVidFile.name)
    frame_width = int(vidCap.get(cv2.CAP_PROP_FRAME_WIDTH)) 
    frame_height = int(vidCap.get(cv2.CAP_PROP_FRAME_HEIGHT)) 
    fps = vidCap.get(cv2.CAP_PROP_FPS) or 25
    size = (frame_width, frame_height)

    output_path = f'{video_name}_pred.mp4'
    out = cv2.VideoWriter(output_path, cv2.VideoWriter_fourcc(*'mp4v'), fps, size)

    while True:
        success, frame = vidCap.read()
        if not success or frame is None:
            break
        processed_frame, _ = process_frame_drowsiness(frame)
        out.write(processed_frame)

    vidCap.release()
    out.release()
    try:
        os.unlink(tempVidFile.name)
    except Exception:
        pass
    return output_path


