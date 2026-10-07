from model import load_saved_model, predict
import numpy as np
import cv2
import tempfile
import mediapipe as mp

model = load_saved_model(load_last=True)

mp_facemesh = mp.solutions.face_mesh
face_mesh_detector = mp_facemesh.FaceMesh(
    static_image_mode=False,
    max_num_faces=1,
    refine_landmarks=True,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

# Landmark Indices
LEFT_EYE = [362, 385, 387, 263, 373, 380]   # [p1, p2, p3, p4, p5, p6]
RIGHT_EYE = [33, 160, 158, 133, 153, 144]   # [p1, p2, p3, p4, p5, p6]
MOUTH = [61, 291, 13, 14, 78, 308]           # corners: 61, 291; upper/lower inner: 13, 14

# State tracking for duration thresholding
eye_closed_counter = 0
yawn_counter = 0

EYE_CLOSED_THRESHOLD = 0.22
YAWN_THRESHOLD = 0.60
FRAMES_EYE_CLOSED = 15
FRAMES_YAWNING = 15

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

def process_frame_drowsiness(img):
    global eye_closed_counter, yawn_counter
    
    h, w = img.shape[:2]
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    results = face_mesh_detector.process(img_rgb)
    
    status = "Active"
    is_drowsy = False
    
    if results.multi_face_landmarks:
        face_landmarks = results.multi_face_landmarks[0].landmark
        
        left_ear = calculate_ear(face_landmarks, LEFT_EYE, w, h)
        right_ear = calculate_ear(face_landmarks, RIGHT_EYE, w, h)
        avg_ear = (left_ear + right_ear) / 2.0
        
        mar = calculate_mar(face_landmarks, MOUTH, w, h)
        
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

        # Trigger alerts
        if eye_closed_counter >= FRAMES_EYE_CLOSED:
            status = "DROWSY! (Eyes Closed)"
            is_drowsy = True
        elif yawn_counter >= FRAMES_YAWNING:
            status = "DROWSY! (Yawning)"
            is_drowsy = True
            
        # Telemetry overlay
        ear_text = f"EAR: {avg_ear:.2f}"
        mar_text = f"MAR: {mar:.2f}"
        cv2.putText(img, ear_text, (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
        cv2.putText(img, mar_text, (20, 70), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (255, 255, 255), 2)
    else:
        # No face detected
        cv2.putText(img, "No Face Detected", (20, 40), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 165, 255), 2)
        eye_closed_counter = 0
        yawn_counter = 0

    # Draw Status banner
    color = (0, 0, 255) if is_drowsy else (0, 255, 0)
    cv2.putText(img, f"Status: {status}", (20, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)
    
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
    tempVidFile = tempfile.NamedTemporaryFile(delete=False)
    tempVidFile.write(video_file.read())

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
    return output_path


