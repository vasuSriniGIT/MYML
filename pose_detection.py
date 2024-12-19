import mediapipe as mp
import math
import cv2

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

def calculate_vertical_distance(p1, p2, p3, p4):
    # Calculate the vertical distance between two lines formed by points (p1, p2) and (p3, p4)
    # Line 1: p1 -> p2 (Shoulder line)
    # Line 2: p3 -> p4 (Stomach line)
    
    # Calculate the slope of both lines
    slope_1 = (p2.y - p1.y) / (p2.x - p1.x) if (p2.x - p1.x) != 0 else 0
    slope_2 = (p4.y - p3.y) / (p4.x - p3.x) if (p4.x - p3.x) != 0 else 0
    
    # Calculate the y-intercepts of both lines
    intercept_1 = p1.y - slope_1 * p1.x if slope_1 != 0 else p1.y
    intercept_2 = p3.y - slope_2 * p3.x if slope_2 != 0 else p3.y
    
    # Calculate the vertical distance between the two lines
    if slope_1 == slope_2:
        return abs(intercept_1 - intercept_2)
    
    # For non-parallel lines, calculate the distance between them
    return abs(slope_1 * p3.x - slope_2 * p1.x + intercept_2 - intercept_1) / math.sqrt(slope_1**2 + 1)

def check_pushup_position(landmarks, frame_idx, distance_threshold = 0.05):
    # Get key landmarks for push-up (Shoulder, Elbow, Wrist)
    left_shoulder = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER]
    left_elbow = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW]
    left_wrist = landmarks[mp.solutions.pose.PoseLandmark.LEFT_WRIST]
    
    right_shoulder = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER]
    right_elbow = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_ELBOW]
    right_wrist = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_WRIST]
    
    # Calculate the angle between shoulder, elbow, and wrist for both arms
    left_angle = calculate_angle(left_shoulder, left_elbow, left_wrist)
    right_angle = calculate_angle(right_shoulder, right_elbow, right_wrist)
    
    # Calculate the vertical distance between the shoulder line and the stomach line
    stomach_line_distance = calculate_vertical_distance(left_shoulder, right_shoulder, landmarks[mp.solutions.pose.PoseLandmark.LEFT_HIP], landmarks[mp.solutions.pose.PoseLandmark.RIGHT_HIP])
    
    # Define thresholds for detecting push-up positions
    DOWN_POSITION_LEFT = 270  # Chest close to the ground (down position)
    UP_POSITION_LEFT = 180  # Arms fully extended (up position)
    DOWN_POSITION_RIGHT = 90  # Chest close to the ground (down position)
    UP_POSITION_RIGHT = 180 # Arms fully extended (up position)
    # Print the diagnostic information every 50th frame
    if frame_idx % 50 == 0:
        print(f"Frame {frame_idx}:")
        print(f"Left Angle: {left_angle:.2f}°")
        print(f"Right Angle: {right_angle:.2f}°")
        print(f"Stomach Line Distance: {stomach_line_distance:.2f} pixels")
    
    # Check for down position based on angle and stomach line distance
    if ((left_angle < DOWN_POSITION_LEFT + 60 or left_angle > DOWN_POSITION_LEFT-5) and
         (right_angle < DOWN_POSITION_RIGHT + 5 and right_angle > DOWN_POSITION_RIGHT-70)
        and stomach_line_distance < distance_threshold):
        if frame_idx % 50 == 0:
            print(f"Position: Down")
        return 'down'
    
    # Check for up position based on angle
    elif (left_angle > UP_POSITION_LEFT -30 and left_angle < UP_POSITION_LEFT+10 and 
          right_angle > UP_POSITION_RIGHT-30 and right_angle < UP_POSITION_RIGHT+10 ):
        if frame_idx % 50 == 0:
            print(f"Position: Up")
        return 'up'
    
    # Mid position if the angles are between up and down
    else:
        if frame_idx % 50 == 0:
            print(f"Position: Mid")
        return 'mid'
