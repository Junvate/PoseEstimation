import cv2
import mediapipe as mp
import math
import numpy as np

# 初始化 MediaPipe 模块
mp_pose = mp.solutions.pose  # 姿态检测模块
mp_hands = mp.solutions.hands  # 手部检测模块
mp_drawing = mp.solutions.drawing_utils  # 绘图工具

# 创建姿态和手部检测对象
pose = mp_pose.Pose(min_detection_confidence=0.5, min_tracking_confidence=0.5)
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=2, 
                       min_detection_confidence=0.75, min_tracking_confidence=0.75)

# 计算二维向量夹角
def vector_2d_angle(v1, v2):
    '''
    计算两个二维向量的夹角（度）
    '''
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

# 计算手部手指角度
def hand_angle(hand_):
    '''
    根据手部关键点计算五个手指的关节角度
    '''
    angle_list = []
    # 大拇指角度
    angle = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[2][0])), (int(hand_[0][1]) - int(hand_[2][1]))),
        ((int(hand_[3][0]) - int(hand_[4][0])), (int(hand_[3][1]) - int(hand_[4][1])))
    )
    angle_list.append(angle)
    # 食指角度
    angle = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[6][0])), (int(hand_[0][1]) - int(hand_[6][1]))),
        ((int(hand_[7][0]) - int(hand_[8][0])), (int(hand_[7][1]) - int(hand_[8][1])))
    )
    angle_list.append(angle)
    # 中指角度
    angle = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[10][0])), (int(hand_[0][1]) - int(hand_[10][1]))),
        ((int(hand_[11][0]) - int(hand_[12][0])), (int(hand_[11][1]) - int(hand_[12][1])))
    )
    angle_list.append(angle)
    # 无名指角度
    angle = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[14][0])), (int(hand_[0][1]) - int(hand_[14][1]))),
        ((int(hand_[15][0]) - int(hand_[16][0])), (int(hand_[15][1]) - int(hand_[16][1])))
    )
    angle_list.append(angle)
    # 小拇指角度
    angle = vector_2d_angle(
        ((int(hand_[0][0]) - int(hand_[18][0])), (int(hand_[0][1]) - int(hand_[18][1]))),
        ((int(hand_[19][0]) - int(hand_[20][0])), (int(hand_[19][1]) - int(hand_[20][1])))
    )
    angle_list.append(angle)
    return angle_list

# 手势识别
def h_gesture(angle_list):
    '''
    根据手指角度识别手势
    '''
    thr_angle = 65.0
    thr_angle_thumb = 53.0
    thr_angle_s = 49.0
    gesture_str = None
    if 65535.0 not in angle_list:
        if (angle_list[0] > thr_angle_thumb) and all(angle > thr_angle for angle in angle_list[1:]):
            gesture_str = "fist"
        elif all(angle < thr_angle_s for angle in angle_list):
            gesture_str = "five"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and all(angle > thr_angle for angle in angle_list[2:]):
            gesture_str = "gun"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] < thr_angle_s) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
            gesture_str = "love"
        elif (angle_list[0] > 5) and (angle_list[1] < thr_angle_s) and all(angle > thr_angle for angle in angle_list[2:]):
            gesture_str = "one"
        elif (angle_list[0] < thr_angle_s) and (angle_list[1] > thr_angle) and (angle_list[2] > thr_angle) and (angle_list[3] > thr_angle) and (angle_list[4] < thr_angle_s):
            gesture_str = "six"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle_s) and (angle_list[4] > thr_angle):
            gesture_str = "three"
        elif (angle_list[0] < thr_angle_s) and all(angle > thr_angle for angle in angle_list[1:]):
            gesture_str = "thumbUp"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] > thr_angle) and (angle_list[4] > thr_angle):
            gesture_str = "two"
        elif (angle_list[0] > thr_angle_thumb) and (angle_list[1] < thr_angle_s) and (angle_list[2] < thr_angle_s) and (angle_list[3] < thr_angle) and (angle_list[4] < thr_angle):
            gesture_str = "four"
    return gesture_str


# 从姿态检测结果提取关键点
def extract_keypoints(results):
    """
    从MediaPipe姿态检测结果中提取关键点坐标
    返回一个包含关键点坐标的数组
    """
    keypoints = []
    if results.pose_landmarks:
        for i in range(33):  # MediaPipe Pose模型有33个关键点
            landmark = results.pose_landmarks.landmark[i]
            keypoints.append(np.array([landmark.x, landmark.y, landmark.z]))
    return keypoints

# 姿态识别函数
def get_pos(keypoints):
    """
    根据关键点识别姿态
    返回识别的姿态字符串
    """
    if len(keypoints) < 17:  # 确保有足够的关键点
        return "UNKNOWN"
    
    str_pose = ""
    # 计算左臂与水平方向的夹角
    keypoints = np.array(keypoints)

    angle_left_arm = vector_2d_angle(
        (keypoints[12][0] - keypoints[11][0] ,keypoints[12][1] - keypoints[11][1]),
        (keypoints[13][0] - keypoints[11][0] ,keypoints[13][1] - keypoints[11][1])
    )
    

    angle_right_arm  = vector_2d_angle(
        (keypoints[12][0] - keypoints[11][0] ,keypoints[12][1] - keypoints[11][1]),
        (keypoints[14][0] - keypoints[12][0] ,keypoints[14][1] - keypoints[12][1])
    )
    
    angle_left_elbow = vector_2d_angle(
        (keypoints[11][0] - keypoints[13][0] ,keypoints[11][1] - keypoints[13][1]),
        (keypoints[13][0] - keypoints[15][0] ,keypoints[13][1] - keypoints[15][1])
    )
    
    angle_right_elbow = vector_2d_angle(
        (keypoints[12][0] - keypoints[14][0] ,keypoints[12][1] - keypoints[14][1]),
        (keypoints[14][0] - keypoints[16][0] ,keypoints[14][1] - keypoints[16][1])
    )
    
# 判断姿态
    # First, check whether arms are raised by comparing y-coordinates
    left_shoulder_y = keypoints[11][1]
    right_shoulder_y = keypoints[12][1]
    left_wrist_y = keypoints[15][1]
    right_wrist_y = keypoints[16][1]
    
    left_arm_raised = left_wrist_y < left_shoulder_y
    right_arm_raised = right_wrist_y < right_shoulder_y
    
    # Now determine the pose based on arm positions and angles
    if left_arm_raised and not right_arm_raised:
        str_pose = "RIGHT_UP"  # Right arm up (from camera perspective)
    elif right_arm_raised and not left_arm_raised:
        str_pose = "LEFT_UP"  # Left arm up (from camera perspective)
    elif left_arm_raised and right_arm_raised:
        str_pose = "ALL_HANDS_UP"  # Both arms up
        # Check if arms are bent to form triangle
        if angle_left_elbow < 130 and angle_right_elbow < 130:
            str_pose = "TRIANGLE"  # Arms bent forming triangle
    else:
        str_pose = "NORMAL"  # Arms down or neutral position
        # Check if hands are on hips (akimbo)
        # if angle_left_elbow < 100 and angle_right_elbow < 100:
        #     str_pose = "AKIMBO"  # Hands on hips
    
    return str_pose

# 修改主函数
def main():
    cap = cv2.VideoCapture(0)  # 打开默认摄像头
    if not cap.isOpened():
        print("Error: Cannot open camera")
        return
    
    # 获取原始摄像头分辨率
    original_width = int(cap.get(cv2.CAP_PROP_FRAME_WIDTH))
    original_height = int(cap.get(cv2.CAP_PROP_FRAME_HEIGHT))
    
    # 设置放大比例
    scale_factor = 1.5
    display_width = int(original_width * scale_factor)
    display_height = int(original_height * scale_factor)

    while True:
        ret, frame = cap.read()  # 读取帧
        if not ret:
            print("Error: Cannot read frame")
            break

        frame = cv2.flip(frame, 1)  # 水平翻转帧
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)  # 转换为 RGB

        # 姿态检测
        pose_results = pose.process(frame_rgb)
        
        # 姿态识别
        if pose_results.pose_landmarks:
            keypoints = extract_keypoints(pose_results)
            pose_str = get_pos(keypoints)
            
            # 绘制姿态关键点
            mp_drawing.draw_landmarks(frame, pose_results.pose_landmarks, mp_pose.POSE_CONNECTIONS)
            
            # 在屏幕上方显示姿态信息
            cv2.putText(frame, f"current pose: {pose_str}", (10, 30), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 255, 0), 2)

        # 手部检测
        hand_results = hands.process(frame_rgb)
        if hand_results.multi_hand_landmarks:
            for idx, hand_landmarks in enumerate(hand_results.multi_hand_landmarks):
                mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)
                hand_local = []
                for i in range(21):
                    x = hand_landmarks.landmark[i].x * frame.shape[1]
                    y = hand_landmarks.landmark[i].y * frame.shape[0]
                    hand_local.append((x, y))
                if hand_local:
                    angle_list = hand_angle(hand_local)
                    gesture_str = h_gesture(angle_list)
                    if gesture_str:
                        # 动态标签位置，基于手腕坐标
                        wrist_x, wrist_y = int(hand_local[0][0]), int(hand_local[0][1])
                        cv2.putText(frame, gesture_str, (wrist_x, wrist_y + 30 * (idx + 1)), 
                                    cv2.FONT_HERSHEY_SIMPLEX, 1.0, (0, 0, 255), 2)

        # 调整显示窗口大小，保持原有比例
        display_frame = cv2.resize(frame, (display_width, display_height))
        
        # 创建一个带有标题的窗口，允许调整大小
        cv2.namedWindow('Pose and Hand Gesture Detection', cv2.WINDOW_NORMAL)
        cv2.imshow('Pose and Hand Gesture Detection', display_frame)  # 显示放大后的帧
        
        key = cv2.waitKey(1) & 0xFF
        if key == ord('q') or key == 27:  # 按 'q' 或 Esc 退出
            break

    cap.release()  # 释放摄像头
    cv2.destroyAllWindows()  # 关闭窗口

if __name__ == '__main__':
    main()