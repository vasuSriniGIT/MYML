# pose_detection.py

import mediapipe as mp
import cv2
import math

def detect_pose(frame, pose):
    # Convert the frame to RGB (since Mediapipe requires RGB images)
    rgb_frame = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = pose.process(rgb_frame)
    return results

def draw_pose_landmarks(frame, results):
    if results.pose_landmarks:
        mp.solutions.drawing_utils.draw_landmarks(
            frame, results.pose_landmarks, mp.solutions.pose.POSE_CONNECTIONS)

def calculate_angle(a, b, c):
    # Calculate the angle between three points
    a = [a.x, a.y]
    b = [b.x, b.y]
    c = [c.x, c.y]
    
    ab = [a[0] - b[0], a[1] - b[1]]
    bc = [c[0] - b[0], c[1] - b[1]]
    
    # Calculate the angle between vectors ab and bc
    angle = math.degrees(math.atan2(bc[1], bc[0]) - math.atan2(ab[1], ab[0]))
    if angle < 0:
        angle += 360
    return angle

def check_pushup_position(landmarks):
    # Get key landmarks for push-up (Shoulder, Elbow, Wrist)
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    
    # Calculate the angle between shoulder, elbow, and wrist
    angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    
    # Define thresholds for detecting push-up positions
    DOWN_POSITION = 90  # Chest close to the ground (down position)
    UP_POSITION = 160  # Arms fully extended (up position)
    
    # Return whether the person is in the down or up position
    if angle < DOWN_POSITION:
        return 'down'
    elif angle > UP_POSITION:
        return 'up'
    else:
        return 'mid'
