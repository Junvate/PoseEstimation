import cv2
import numpy as np
import mediapipe as mp
from mediapipe.tasks import python
from mediapipe.tasks.python import vision
from mediapipe.framework.formats import landmark_pb2

# Path to the pose landmarker model (update this to your model file location)
model_path = "task/pose_landmarker_full.task"

# Video file paths
video_input_path = "video1.mp4"  # Input video file
video_output_path = "output_video.mp4"  # Output video file

# Configuration parameters
num_poses = 10
min_pose_detection_confidence = 0.5
min_pose_presence_confidence = 0.5
min_tracking_confidence = 0.5

# Function to draw landmarks on the image
def draw_landmarks_on_image(rgb_image, detection_result):
    if detection_result is None or not detection_result.pose_landmarks:
        return rgb_image  # Return original image if no landmarks detected

    pose_landmarks_list = detection_result.pose_landmarks
    annotated_image = np.copy(rgb_image)

    for pose_landmarks in pose_landmarks_list:
        # Convert landmarks to a format suitable for drawing
        pose_landmarks_proto = landmark_pb2.NormalizedLandmarkList()
        pose_landmarks_proto.landmark.extend([
            landmark_pb2.NormalizedLandmark(x=landmark.x, y=landmark.y, z=landmark.z)
            for landmark in pose_landmarks
        ])
        # Draw landmarks and connections
        mp.solutions.drawing_utils.draw_landmarks(
            annotated_image,
            pose_landmarks_proto,
            mp.solutions.pose.POSE_CONNECTIONS,
            mp.solutions.drawing_styles.get_default_pose_landmarks_style()
        )
    return annotated_image

# Configure PoseLandmarker options (no result_callback)
base_options = python.BaseOptions(model_asset_path=model_path)
options = vision.PoseLandmarkerOptions(
    base_options=base_options,
    running_mode=vision.RunningMode.VIDEO,  # Set to video mode
    num_poses=num_poses,
    min_pose_detection_confidence=min_pose_detection_confidence,
    min_pose_presence_confidence=min_pose_presence_confidence,
    min_tracking_confidence=min_tracking_confidence,
    output_segmentation_masks=False
)

# Process the video
with vision.PoseLandmarker.create_from_options(options) as landmarker:
    # Open the input video
    cap = cv2.VideoCapture(video_input_path)
    if not cap.isOpened():
        print("Error: Cannot open video file")
        exit()

    # Get video properties
    frame_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    frame_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    fps = cap.get(cv2.CAP_PROP_FPS)

    # Set up video writer for output
    fourcc = cv2.VideoWriter_fourcc(*'mp4v')
    out = cv2.VideoWriter(video_output_path, fourcc, fps, (frame_width, frame_height))

    frame_count = 0
    while cap.isOpened():
        success, image = cap.read()
        if not success:
            print("End of video")
            break

        # Convert the frame to a MediaPipe Image
        rgb_image = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
        mp_image = mp.Image(image_format=mp.ImageFormat.SRGB, data=rgb_image)

        # Calculate timestamp for the frame (in milliseconds)
        timestamp_ms = int(1000 * frame_count / fps)

        # Detect poses in the video frame
        detection_result = landmarker.detect_for_video(mp_image, timestamp_ms)

        # Draw landmarks on the frame
        annotated_image = draw_landmarks_on_image(rgb_image, detection_result)
        output_frame = cv2.cvtColor(annotated_image, cv2.COLOR_RGB2BGR)

        # Write the processed frame to the output video
        out.write(output_frame)

        # Display the processed frame (optional)
        cv2.imshow("MediaPipe Pose Landmark", output_frame)
        if cv2.waitKey(1) & 0xFF == ord('q'):
            break

        frame_count += 1

    # Clean up
    cap.release()
    out.release()
    cv2.destroyAllWindows()