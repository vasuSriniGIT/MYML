import sys
import traceback
import cv2
import os
import mediapipe as mp
from video_utils import load_video, get_video_properties, fast_forward
from pose_detection import detect_pose, draw_pose_landmarks, check_pushup_position
from config import VIDEO_PATH, FAST_FORWARD_PERCENTAGE


def save_frame_with_pose(frame, output_folder, frame_count):
    """
    Save a single frame with pose landmarks drawn on it.
    
    Args:
        frame: The frame to save.
        output_folder: The folder where images will be saved.
        frame_count: The current frame count (used for naming files).
    """
    if not os.path.exists(output_folder):
        os.makedirs(output_folder)

    frame_path = os.path.join(output_folder, f"frame_{frame_count:04d}.jpg")
    cv2.imwrite(frame_path, frame)

def main():
    try:
        # Load video
        cap = load_video(VIDEO_PATH)
        
        # Get video properties
        frame_count, fps, frame_width, frame_height = get_video_properties(cap)
        print(f"Total Frames: {frame_count}")
        print(f"FPS: {fps}")
        print(f"Frame Width: {frame_width}")
        print(f"Frame Height: {frame_height}")
        
        # Fast forward video by the specified percentage
        fast_forward(cap, FAST_FORWARD_PERCENTAGE)
        
        # Initialize Pose Detection
        pushup_count = 0
        previous_position_2 = 'mid'
        previous_position_1 = 'up'  # Initial position (assume they start at the top)
        output_folder = os.path.join("data", os.path.splitext(os.path.basename(VIDEO_PATH))[0])
        frame_interval = 50  # Save every 50th frame
        
        with mp.solutions.pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
            frame_idx = 0
            while cap.isOpened():
                ret, frame = cap.read()
                if not ret:
                    print("Failed to read frame, stopping video.")
                    break

                # Detect pose and draw landmarks
                results = detect_pose(frame, pose)
                draw_pose_landmarks(frame, results)

                # Save frames with landmarks every 50th frame
                if frame_idx % frame_interval == 0:
                    save_frame_with_pose(frame, output_folder, frame_idx)

                # Get the push-up position (down, up, or mid)
                if results.pose_landmarks:
                    position = check_pushup_position(results.pose_landmarks.landmark, frame_idx)
                    # Count push-ups (only count when transitioning from down to up or up to down)
                    if position == 'up' and (previous_position_1 == 'mid' or previous_position_2 == 'down'):
                        pushup_count += 1
                        print(f"Push-up Count: {pushup_count}")
                    previous_position = position
                    if position != previous_position_1:
                        previous_position_2 = previous_position_1
                        previous_position_1 = position

                # Display push-up count and position on the frame using one cv2.putText()
                display_text = ' | '.join(['Push-ups: ' + str(pushup_count), 'Position: ' + previous_position])

                # Display the combined text on the frame at the desired location
                cv2.putText(frame, display_text, (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
                
                # Show the frame with detected pose landmarks and push-up count
                cv2.imshow('Pose Detection', frame)

                # Check for 'q' key press to exit
                if cv2.waitKey(1) & 0xFF == ord('q'):
                    break

                frame_idx += 1

        # Release the video capture object and close OpenCV windows
        cap.release()
        cv2.destroyAllWindows()

    except Exception as e:
        print(f"Error occurred: {e}")
        traceback.print_exc()

if __name__ == "__main__":
    main()
