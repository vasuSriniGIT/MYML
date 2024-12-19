# main.py

import cv2
from video_utils import load_video, get_video_properties, fast_forward
from pose_detection import detect_pose, draw_pose_landmarks, check_pushup_position
from config import VIDEO_PATH, FAST_FORWARD_PERCENTAGE
import mediapipe as mp

def main():
    # Load video
    cap = load_video(VIDEO_PATH)
    
    # Get video properties
    frame_count, fps, frame_width, frame_height = get_video_properties(cap)
    print(f"Total Frames: {frame_count}")
    print(f"FPS: {fps}")
    print(f"Frame Width: {frame_width}")
    print(f"Frame Height: {frame_height}")
    
    # Fast forward video
    fast_forward(cap, FAST_FORWARD_PERCENTAGE)
    
    # Initialize Pose Detection
    pushup_count = 0
    previous_position = 'up'  # Initial position (assume they start at the top)
    
    with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break

            # Detect pose and draw landmarks
            results = detect_pose(frame, pose)
            draw_pose_landmarks(frame, results)

            # Get the push-up position (down, up, or mid)
            if results.pose_landmarks:
                position = check_pushup_position(results.pose_landmarks.landmark)

                # Count push-ups (only count when transitioning from down to up or up to down)
                if position == 'up' and previous_position == 'down':
                    pushup_count += 1
                    print(f"Push-up Count: {pushup_count}")
                previous_position = position

            # Display push-up count on the frame
            cv2.putText(frame, f'Push-ups: {pushup_count}', (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)

            # Show the frame with detected pose landmarks and push-up count
            cv2.imshow('Pose Detection', frame)

            # Check for 'q' key press to exit
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    # Release the video capture object and close OpenCV windows
    cap.release()
    cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
