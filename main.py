import cv2
import mediapipe as mp

mpPose = mp.solutions.pose  # 导入姿态检测模块
pose_mode = mpPose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)  # 创建姿态检测对象并设置参数
mpDraw = mp.solutions.drawing_utils  # 导入绘图工具


cap = cv2.VideoCapture(0)  # 使用摄像头捕获视频，参数0表示使用默认摄像头
 
while True:
    success, img = cap.read()  # 读取视频帧
    img = cv2.flip(img, 1)  # 翻转图像，使画面更加自然
    results = pose_mode.process(img)  # 对图像进行姿态检测
 
    if results.pose_landmarks:
        mpDraw.draw_landmarks(img, results.pose_landmarks, mpPose.POSE_CONNECTIONS)  # 绘制姿态关键点和连接线
 
    cv2.imshow("img", img)  # 显示图像
    if cv2.waitKey(1) & 0xFF == ord("q"):  # 按下'q'键退出循环
        break
 
cap.release()  # 释放摄像头资源
cv2.destroyAllWindows()  # 关闭所有OpenCV窗口