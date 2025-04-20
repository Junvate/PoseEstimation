import cv2
import mediapipe as mp
import logging

# 屏蔽 TensorFlow 警告
logging.getLogger('tensorflow').setLevel(logging.ERROR)

# 初始化 MediaPipe
mp_pose = mp.solutions.pose
mp_drawing = mp.solutions.drawing_utils

# 打开视频
cap = cv2.VideoCapture('./video1.mp4')
if not cap.isOpened():
    print("错误：无法打开视频文件。")
    exit()

# 获取视频属性
width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
fps = cap.get(cv2.CAP_PROP_FPS)
# 确保分辨率为偶数
width = width - (width % 2)
height = height - (height % 2)
print(f"分辨率: {width}x{height}, 帧率: {fps}")

# 初始化视频写入器
fourcc = cv2.VideoWriter_fourcc(*'H264')  # 尝试 H264
out = cv2.VideoWriter('output.mp4', fourcc, fps, (width, height))
if not out.isOpened():
    print("错误：无法创建输出视频文件，尝试使用 XVID 和 .avi")
    fourcc = cv2.VideoWriter_fourcc(*'XVID')
    out = cv2.VideoWriter('output.avi', fourcc, fps, (width, height))
    if not out.isOpened():
        print("错误：仍然无法创建输出视频文件。")
        cap.release()
        exit()

# 初始化姿态检测器
with mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5) as pose:
    try:
        while cap.isOpened():
            ret, frame = cap.read()
            if not ret:
                break
            frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = pose.process(frame_rgb)
            if results.pose_landmarks:
                mp_drawing.draw_landmarks(frame, results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            out.write(frame)
            cv2.imshow('Pose Detection', frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break
    except Exception as e:
        print(f"发生错误: {e}")
    finally:
        cap.release()
        out.release()
        cv2.destroyAllWindows()