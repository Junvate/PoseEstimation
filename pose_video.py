import cv2
import mediapipe as mp
import math
import numpy as np
import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Initialize MediaPipe modules
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# Suppress TensorFlow warnings
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# Calculate angle between two 2D vectors
def vector_2d_angle(v1, v2):
    """
    Calculate the angle (in degrees) between two 2D vectors.
    """
    v1_x, v1_y = v1
    v2_x, v2_y = v2
    try:
        angle = math.degrees(math.acos((v1_x * v2_x + v1_y * v2_y) / 
                                     (((v1_x**2 + v1_y**2)**0.5) * ((v2_x**2 + v2_y**2)**0.5))))
    except:
        angle = 65535.0
    if angle > 180.0:
        angle = 65535.0
    return angle

# Extract keypoints from pose landmarks
def extract_keypoints(pose_landmarks):
    """
    Extract keypoints coordinates from MediaPipe pose landmarks.
    Returns a list of keypoints as numpy arrays.
    """
    keypoints = []
    if pose_landmarks:
        for i in range(33):  # MediaPipe Pose has 33 keypoints
            landmark = pose_landmarks.landmark[i]
            keypoints.append(np.array([landmark.x, landmark.y, landmark.z]))
    return keypoints

# Recognize pose based on keypoints
def get_pos(keypoints):
    """
    Recognize pose based on keypoints.
    Returns the pose string.
    """
    if len(keypoints) < 17:
        return "UNKNOWN"
    
    str_pose = ""
    keypoints = np.array(keypoints)

    # Calculate angles for arms and elbows
    angle_left_arm = vector_2d_angle(
        (keypoints[12][0] - keypoints[11][0], keypoints[12][1] - keypoints[11][1]),
        (keypoints[13][0] - keypoints[11][0], keypoints[13][1] - keypoints[11][1])
    )
    angle_right_arm = vector_2d_angle(
        (keypoints[12][0] - keypoints[11][0], keypoints[12][1] - keypoints[11][1]),
        (keypoints[14][0] - keypoints[12][0], keypoints[14][1] - keypoints[12][1])
    )
    angle_left_elbow = vector_2d_angle(
        (keypoints[11][0] - keypoints[13][0], keypoints[11][1] - keypoints[13][1]),
        (keypoints[13][0] - keypoints[15][0], keypoints[13][1] - keypoints[15][1])
    )
    angle_right_elbow = vector_2d_angle(
        (keypoints[12][0] - keypoints[14][0], keypoints[12][1] - keypoints[14][1]),
        (keypoints[14][0] - keypoints[16][0], keypoints[14][1] - keypoints[16][1])
    )
    
    # Determine pose
    left_shoulder_y = keypoints[11][1]
    right_shoulder_y = keypoints[12][1]
    left_wrist_y = keypoints[15][1]
    right_wrist_y = keypoints[16][1]
    
    left_arm_raised = left_wrist_y < left_shoulder_y
    right_arm_raised = right_wrist_y < right_shoulder_y
    
    if left_arm_raised and not right_arm_raised:
        str_pose = "RIGHT_UP"
    elif right_arm_raised and not left_arm_raised:
        str_pose = "LEFT_UP"
    elif left_arm_raised and right_arm_raised:
        str_pose = "ALL_HANDS_UP"
        if angle_left_elbow < 130 and angle_right_elbow < 130:
            str_pose = "TRIANGLE"
    else:
        str_pose = "NORMAL"
    
    return str_pose

# Get head position for label placement
def get_head_position(pose_landmarks, frame_shape):
    """
    Get head position (nose) for placing the pose label.
    Returns pixel coordinates (x, y).
    """
    if pose_landmarks:
        nose = pose_landmarks.landmark[0]  # Nose keypoint
        height, width = frame_shape[:2]
        x = int(nose.x * width)
        y = int(nose.y * height) - 50  # Move 50 pixels above head
        return x, y
    return None

def main():
    # Open video file
    video_path = './video1.mp4'
    cap = cv2.VideoCapture(video_path)
    
    if not cap.isOpened():
        logger.error(f"Cannot open video file {video_path}")
        return
    
    # Get video properties
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)
    
    # Ensure even resolution
    original_width = original_width - (original_width % 2)
    original_height = original_height - (original_height % 2)
    
    logger.info(f"Video Resolution: {original_width}x{original_height}, FPS: {fps}")
    
    # Initialize video writer
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter('output_pose.mp4', fourcc, fps, (original_width, original_height))
    
    if not out.isOpened():
        logger.warning("Cannot create output video file, trying XVID and .avi")
        fourcc = cv2.VideoWriter_fourcc(*'XVID')
        out = cv2.VideoWriter('output_pose.avi', fourcc, fps, (original_width, original_height))
        if not out.isOpened():
            logger.error("Still cannot create output video file.")
            cap.release()
            return
    
    # Set display scale
    scale_factor = 1.0
    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)
    
    # Create Pose detector with multi-person support
    try:
        pose = mp_pose.Pose(
            min_detection_confidence=0.5,
            min_tracking_confidence=0.5,
            num_poses=5  # Support up to 5 people
        )
    except TypeError as e:
        logger.error(f"Failed to initialize Pose: {e}")
        logger.error("Ensure MediaPipe version supports 'max_num_poses'. Current version: %s", mp.__version__)
        cap.release()
        return
    
    frame_count = 0
    total_frames = int(cap.get(cv2.CAP_PROP_FRAME_COUNT))
    
    while True:
        ret, frame = cap.read()
        if not ret:
            logger.info("End of video or error reading frame")
            break
        
        # Process frame
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        pose_results = pose.process(frame_rgb)
        
        # Process multiple pose detections
        if pose_results.pose_landmarks:
            for person_idx, person_landmarks in enumerate(pose_results.pose_landmarks):
                keypoints = extract_keypoints(person_landmarks)
                pose_str = get_pos(keypoints)
                
                # Draw landmarks
                mp_drawing.draw_landmarks(
                    frame,
                    person_landmarks,
                    mp_pose.POSE_CONNECTIONS,
                    mp_drawing.DrawingSpec(color=(0, 255, 0), thickness=2, circle_radius=2),
                    mp_drawing.DrawingSpec(color=(0, 0, 255), thickness=2)
                )
                
                # Draw pose label above head
                head_pos = get_head_position(person_landmarks, frame.shape)
                if head_pos:
                    x, y = head_pos
                    cv2.putText(
                        frame,
                        f"Pose: {pose_str}",
                        (x, y),
                        cv2.FONT_HERSHEY_SIMPLEX,
                        0.7,
                        (0, 255, 0),
                        2
                    )
        
        # Write frame to output video
        out.write(frame)
        
        # Resize for display
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # Show progress
        frame_count += 1
        progress = (frame_count / total_frames) * 100
        print(f"Processing: {progress:.1f}% ({frame_count}/{total_frames})", end='\r')
        
        # Display frame
        cv2.namedWindow('Multi-Pose Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Multi-Pose Detection', display_frame)
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:
            break
    
    logger.info("Processing complete")
    cap.release()
    out.release()
    cv2.destroyAllWindows()
    pose.close()

if __name__ == '__main__':
    main()