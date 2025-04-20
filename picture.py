import cv2
import mediapipe as mp
import requests
import numpy as np
from io import BytesIO

# Initialize MediaPipe drawing tools and pose detection module
mp_drawing = mp.solutions.drawing_utils
mp_pose = mp.solutions.pose

# Set drawing styles for keypoints and lines
DrawingSpec_point = mp_drawing.DrawingSpec((0, 255, 0), 2, 2)
DrawingSpec_line = mp_drawing.DrawingSpec((0, 0, 255), 2, 2)

# Initialize pose detection object for static image mode
pose = mp_pose.Pose(static_image_mode=True)

# Download image from URL
url = 'http://images.cocodataset.org/test2017/000000032755.jpg'
response = requests.get(url)
if response.status_code != 200:
    print("Failed to download image")
    exit()

# Convert image data to OpenCV format
image_data = BytesIO(response.content)
image_array = np.asarray(bytearray(image_data.read()), dtype=np.uint8)
image = cv2.imdecode(image_array, cv2.IMREAD_COLOR)

# Check if image loaded successfully
if image is None:
    print("Failed to load image")
    exit()

# Get image dimensions
image_height, image_width, _ = image.shape

# Convert BGR image to RGB
image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

# Process RGB image to get pose detection results
results = pose.process(image_rgb)

# Draw pose landmarks and connections on the image
if results.pose_landmarks:
    mp_drawing.draw_landmarks(
        image, results.pose_landmarks, mp_pose.POSE_CONNECTIONS,
        DrawingSpec_point, DrawingSpec_line
    )
    # Print nose coordinates
    print(
        f'Nose coordinates: ('
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].x * image_width}, '
        f'{results.pose_landmarks.landmark[mp_pose.PoseLandmark.NOSE].y * image_height})'
    )

# Save the output image
cv2.imwrite('image-pose.jpg', image)

# Release pose detection object
pose.close()