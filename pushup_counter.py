def analyze_pushup_pose(results):
    """Analyze the pose landmarks to determine push-up state."""
    if not results.pose_landmarks:
        return None

    # Get the landmark for the nose and shoulders
    landmarks = results.pose_landmarks.landmark

    # Extract shoulder and elbow y-coordinates (relative to frame height)
    left_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_SHOULDER].y
    right_shoulder_y = landmarks[mp.solutions.pose.PoseLandmark.RIGHT_SHOULDER].y
    left_elbow_y = landmarks[mp.solutions.pose.PoseLandmark.LEFT_ELBOW].y

    # Determine the average shoulder and elbow height
    avg_shoulder_y = (left_shoulder_y + right_shoulder_y) / 2

    # Push-up logic (returning a simple state: "up" or "down")
    if left_elbow_y > avg_shoulder_y + 0.05:  # Threshold to detect "down"
        return "down"
    elif left_elbow_y < avg_shoulder_y - 0.05:  # Threshold to detect "up"
        return "up"
    return None

def count_pushups(pushup_state, prev_state, count):
    """Count push-ups based on state transitions."""
    if prev_state == "down" and pushup_state == "up":
        count += 1
    return count
