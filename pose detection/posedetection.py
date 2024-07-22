import cv2
import mediapipe as mp
import numpy as np

mp_pose = mp.solutions.pose
pose = mp_pose.Pose()
mp_drawing = mp.solutions.drawing_utils

def calculate_angle(a, b, c):
    a = np.array(a)
    b = np.array(b)
    c = np.array(c)
    
    ab = a - b
    bc = c - b
    
    dot_product = np.dot(ab, bc)
    norm_ab = np.linalg.norm(ab)
    norm_bc = np.linalg.norm(bc)
    
    cos_angle = dot_product / (norm_ab * norm_bc)
    angle = np.arccos(np.clip(cos_angle, -1.0, 1.0))
    return np.degrees(angle)

cap = cv2.VideoCapture(0)

if not cap.isOpened():
    print("Error: Could not access webcam.")
    exit()

print("Webcam is working. Starting pose detection...")

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        print("Error: Failed to capture image.")
        break

    image_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(image_rgb)
    image_bgr = cv2.cvtColor(image_rgb, cv2.COLOR_RGB2BGR)

    if results.pose_landmarks:
        mp_drawing.draw_landmarks(image_bgr, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
        
        landmarks = results.pose_landmarks.landmark
        h, w, _ = image_bgr.shape
        
        shoulder = (landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].x * w, landmarks[mp_pose.PoseLandmark.LEFT_SHOULDER].y * h)
        elbow = (landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].x * w, landmarks[mp_pose.PoseLandmark.LEFT_ELBOW].y * h)
        wrist = (landmarks[mp_pose.PoseLandmark.LEFT_WRIST].x * w, landmarks[mp_pose.PoseLandmark.LEFT_WRIST].y * h)
        
        angle = calculate_angle(shoulder, elbow, wrist)
        
        cv2.putText(image_bgr, f'Angle: {angle:.2f}', (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
    else:
        print("pose landmarks detected.")
    
    cv2.imshow('Pose Detection', image_bgr)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
