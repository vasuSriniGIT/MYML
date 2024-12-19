### video_utils.py
import cv2

def load_video(video_path):
    cap = cv2.VideoCapture(video_path)
    if not cap.isOpened():
        raise FileNotFoundError(f"Error: Cannot open video file at {video_path}")
    return cap

def get_video_properties(cap):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    return frame_count, fps, frame_width, frame_height

def fast_forward(cap, percentage):
    frame_count = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    target_frame = int(frame_count * percentage / 100)
    cap.set(cv2.CAP_PROP_POS_FRAMES, target_frame)
