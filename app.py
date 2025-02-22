import cv2
import mediapipe as mp
import math
import statistics
import net_init
import numpy as np
#import pyautogui

# # 获取屏幕宽度和高度
# screen_width, screen_height = pyautogui.size()

# # 设置每个窗口的宽度为屏幕的一半，保持高度不变
# camera_width = screen_width // 2
# camera_height = screen_height

# gesture_width = screen_width // 2
# gesture_height = screen_height

video_capture = cv2.VideoCapture(0)

if not video_capture.isOpened():
    print("Cannot access the camera.")
    exit()
else:
    print("Camera OK.")

# 获取视频的宽度和高度
# frame_width = int(video_capture.get(cv2.CAP_PROP_FRAME_WIDTH))
# frame_height = int(video_capture.get(cv2.CAP_PROP_FRAME_HEIGHT))

# # 设置目标区域的宽度和高度（假设你想裁剪为中心区域）
# target_width = 640
# target_height = 480

# # 设置文字参数
# font = cv2.FONT_HERSHEY_SIMPLEX
# font_scale = 1
# color = (255, 255, 255)  # 白色
# thickness = 2

# while True:
#     # 读取摄像头的帧
#     ret, frame = video_capture.read()
#     if not ret:
#         print("Failed to capture image.")
#         break

#     # 裁剪中心区域
#     x_start = (frame_width - target_width) // 2
#     y_start = (frame_height - target_height) // 2
#     cropped_frame = frame[y_start:y_start+target_height, x_start:x_start+target_width]


mediapipe = mp.solutions.hands
mediapipe_hands = mediapipe.Hands(static_image_mode = False, max_num_hands = 1, min_detection_confidence = 0.5, min_tracking_confidence = 0.5)
mp_drawing = mp.solutions.drawing_utils

hand_in_frame = False
hand_position_eval = "OUT_OF_FRAME"

anchor_distances_avg = float(0)

centroid_position_xy_1 = []
centroid_position_xy_2 = []
centroid_calculation_iteration = 1

centroid_position_xy_1 = []
centroid_position_xy_2 = []

hand_centroid_velocity = float(0)

net_testing = False

net_output_translation = "-"
net_output_translation_confidence = "-"

def softmax_translation(softmax_argmax):
    alphabet = ['A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    translation = alphabet[softmax_argmax]

    return translation

def calculate_hand_centroid_velocity(centroid_position_1_x_y, centroid_position_2_x_y):
    distance_between_pos_1_2 = math.sqrt((math.pow(centroid_position_2_x_y[0] - centroid_position_1_x_y[0], 2)) + (math.pow(centroid_position_2_x_y[1] - centroid_position_1_x_y[1], 2)))

    time_velocity = float(1)

    hand_centroid_velocity = distance_between_pos_1_2 / time_velocity

    return hand_centroid_velocity

def calculate_hand_anchor_distances(point1, point2):
    x1 = point1[0]
    y1 = point1[1]
    x2 = point2[0]
    y2 = point2[1]

    distance = math.sqrt((x2 - x1) ** 2 + (y2 - y1) **2)

    return distance   

while video_capture:
    success, original_frame = video_capture.read()

    frame_y_max = original_frame.shape[0]
    frame_x_max = original_frame.shape[1]   

    analytics_frame = original_frame.copy()
    test_frame = original_frame.copy()

    original_frame_rgb = cv2.cvtColor(original_frame, cv2.COLOR_BGR2RGB)

    mediapipe_hands_results = mediapipe_hands.process(original_frame_rgb)

    hand_landmark_x_coordinates = []
    hand_landmark_y_coordinates = []
    
    if mediapipe_hands_results.multi_hand_landmarks:
        hand_in_frame = True

        for hand_landmarks in mediapipe_hands_results.multi_hand_landmarks:            
            landmark_points_color = mp_drawing.DrawingSpec(color = (221, 0, 255), thickness = 4, circle_radius = 1)
            landmark_connections_points_color = mp_drawing.DrawingSpec(color = (221 , 0, 255), thickness = 4, circle_radius = 1)
            mp_drawing.draw_landmarks(test_frame, hand_landmarks, mediapipe.HAND_CONNECTIONS, landmark_points_color, landmark_connections_points_color)             

            for id, lm in enumerate(hand_landmarks.landmark):                
                h, w, c = original_frame.shape  
                hand_landmark_coordinate_x, hand_landmark_coordinate_y = int(lm.x * w), int(lm.y * h)                                      
                hand_landmark_x_coordinates.append(hand_landmark_coordinate_x)
                hand_landmark_y_coordinates.append(hand_landmark_coordinate_y)                      

        hand_bounding_box_x_min = min(hand_landmark_x_coordinates)
        hand_bounding_box_x_max = max(hand_landmark_x_coordinates)
        hand_bounding_box_y_min = min(hand_landmark_y_coordinates)
        hand_bounding_box_y_max = max(hand_landmark_y_coordinates)          

        # BOUNDING BOX AND CENTROID
        cv2.rectangle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 255, 0), 2)            
        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 255, 0), 1)                                    

        # BOUNDING BOX ANCHORS
        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), 5, (255, 0, 0), 10)
        top_middle_anchor_distance = calculate_hand_anchor_distances([int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)],[int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2),int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)])
        
        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), 5, (255, 0, 0), 10)     
        bottom_middle_anchor_distance = calculate_hand_anchor_distances([int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)], [int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)])

        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (255, 0, 0), 10)            
        left_middle_anchor_distance = calculate_hand_anchor_distances([int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)],[int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)])

        cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (255, 0, 0), 10)            
        right_middle_anchor_distance = calculate_hand_anchor_distances([int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)],[int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)])                        

        anchor_distances_list = [top_middle_anchor_distance, bottom_middle_anchor_distance, left_middle_anchor_distance, right_middle_anchor_distance]            

        anchor_distances_avg = float(statistics.mean(anchor_distances_list))

        # HAND POSITION EVALUATION
        if hand_bounding_box_x_min < (hand_bounding_box_x_min + hand_bounding_box_x_max) / 2 - int(150 / 1.25):
            hand_position_eval = "TOO_CLOSE"
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 0, 255), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)                 

        elif hand_bounding_box_x_min > (hand_bounding_box_x_min + hand_bounding_box_x_max) / 2 - int(150/1.25) and anchor_distances_avg < 70:                
            hand_position_eval= "IN_RANGE"
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 255, 0), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 255, 0), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 255, 0), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 255, 0), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 255, 0), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 255, 0), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 255, 0), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 255, 0), 3)                  

        elif hand_bounding_box_x_min > (hand_bounding_box_x_min + hand_bounding_box_x_max) / 2 - int(150/1.25) and anchor_distances_avg > 70:                
            hand_position_eval = "TOO_FAR"  
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 0, 255), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)                           

        else:
            hand_position_eval = "TOO_CLOSE"
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), 5, (0, 0, 255), 10)
            cv2.circle(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), 5, (0, 0, 255), 10)

            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_min)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame, (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int(hand_bounding_box_y_max)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + int(150/1.25)), (0, 0, 255), 3)
            cv2.line(analytics_frame,(int(hand_bounding_box_x_max), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)            
            cv2.line(analytics_frame, (int(hand_bounding_box_x_min), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - int(150/1.25), int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2)), (0, 0, 255), 3)         

        test_frame_x1 = int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) - 150
        test_frame_y1 = int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) - 150
        test_frame_x2 = int((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2) + 150
        test_frame_y2 = int((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2) + 150  

        if test_frame_x1 >= 0 and test_frame_x2 <= frame_x_max and test_frame_y1 >= 0 and test_frame_y2 <= frame_y_max:                   
            forward_frame = test_frame[test_frame_y1:test_frame_y2, test_frame_x1:test_frame_x2]    
            forward_frame_blur_1 = cv2.blur(forward_frame, (5, 5))
            forward_frame_blur_2 = cv2.medianBlur(forward_frame_blur_1, 5)
            forward_frame_blur_3 = cv2.GaussianBlur(forward_frame_blur_2,(5, 5),0)
            forward_frame_blur_4 = cv2.bilateralFilter(forward_frame_blur_3, 9, 75, 75)                        

            forward_frame_hsv = cv2.cvtColor(forward_frame_blur_4, cv2.COLOR_BGR2HSV)
            forward_frame_mask = cv2.inRange(forward_frame_hsv, (135, 100, 20), (160, 255, 255))
            foward_threshold = cv2.threshold(forward_frame_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
            forward_inverse_threshold = cv2.threshold(foward_threshold, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
            forward_inverse_threshold = cv2.flip(forward_inverse_threshold, 1)

            
            # # 在手势窗口中显示黑色背景
            # gesture_window = np.zeros((frame_height, frame_width, 3), dtype=np.uint8)

            #  # 在手势窗口中插入裁剪后的摄像头画面
            # gesture_window[y_start:y_start+target_height, x_start:x_start+target_width] = cropped_frame
            # print("Displaying gesture window...")
            # cv2.imshow('Hand Segmentation', forward_inverse_threshold)
        
            # #Resize gesture window
            # # gesture_frame_resized = cv2.resize(test_frame, (gesture_width, gesture_height))
            # # # 显示手势识别窗口，设置为屏幕左边
            # # cv2.imshow('Gesture', gesture_frame_resized)
            # # cv2.moveWindow('Gesture', 0, 0)  # 左边
            cv2.imshow('Hand Segmentation', forward_inverse_threshold)  
            
        

        if centroid_calculation_iteration == 1:
            centroid_position_xy_1.append(((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2))
            centroid_position_xy_1.append(((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2))
            centroid_calculation_iteration = centroid_calculation_iteration + 1
        elif centroid_calculation_iteration == 2:
            centroid_position_xy_2.append(((hand_bounding_box_x_min + hand_bounding_box_x_max) / 2))
            centroid_position_xy_2.append(((hand_bounding_box_y_min + hand_bounding_box_y_max) / 2))   

            hand_centroid_velocity = float(calculate_hand_centroid_velocity(centroid_position_xy_1, centroid_position_xy_2))
            
            if net_testing == False and hand_in_frame == True and hand_centroid_velocity != 0 and hand_centroid_velocity < 1.5 and hand_position_eval == 'IN_RANGE' and test_frame_x1 >= 0 and test_frame_x2 <= frame_x_max and test_frame_y1 >= 0 and test_frame_y2 <= frame_y_max:             
                net_testing = True

                forward_frame = test_frame[test_frame_y1 : test_frame_y2, test_frame_x1 : test_frame_x2]    
                forward_frame_blur_1 = cv2.blur(forward_frame, (5, 5))
                forward_frame_blur_2 = cv2.medianBlur(forward_frame_blur_1, 5)
                forward_frame_blur_3 = cv2.GaussianBlur(forward_frame_blur_2, (5, 5), 0)
                forward_frame_blur_4 = cv2.bilateralFilter(forward_frame_blur_3, 9, 75, 75)                        

                forward_frame_hsv = cv2.cvtColor(forward_frame_blur_4, cv2.COLOR_BGR2HSV)
                forward_frame_mask = cv2.inRange(forward_frame_hsv,(135, 100, 20), (160, 255, 255))
                foward_threshold = cv2.threshold(forward_frame_mask, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
                forward_inverse_threshold = cv2.threshold(foward_threshold, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
                forward_inverse_threshold_resized = cv2.resize(forward_inverse_threshold, dsize=(28, 28), interpolation = cv2.INTER_AREA)  
        
                net_softmax_output = net_init.trained_forward_propagation(forward_inverse_threshold_resized)
                net_softmax_output_list = net_softmax_output.tolist()

                net_output_translation_confidence = "{:.2%}".format(net_softmax_output_list[np.argmax(net_softmax_output_list)])
                net_output_translation = str(softmax_translation(np.argmax(net_softmax_output)))              
                net_testing = False
                
            centroid_position_xy_1 = []
            centroid_position_xy_2 = []
            centroid_calculation_iteration = 1

    else:
        hand_in_frame = False
        hand_position_eval = "OUT_OF_FRAME"
        anchor_distances_avg = float(0)
        centroid_position_xy_1 = [] 
        centroid_position_xy_2 = []
        hand_landmark_x_coordinates = []
        hand_landmark_y_coordinates = []
        hand_centroid_velocity = float(0)
        centroid_calculation_iteration = 1  
        hand_position_eval = "OUT_OF_FRAME" 
        net_testing = False       
        net_output_translation = "-"  
        net_output_translation_confidence = "-"
      
    analytics_frame = cv2.flip(analytics_frame, 1)


    cv2.putText(analytics_frame, "Hit Q to QUIT!", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 98, 255), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "hand_in_frame: {}".format(str(hand_in_frame)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "hand_position_eval: {}".format(hand_position_eval), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "anchor_distances_avg: {}".format(float(anchor_distances_avg)), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "hand_centroid_velocity: {}".format(hand_centroid_velocity), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "net_testing: {}".format(net_testing), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "net_output_translation: {}".format(net_output_translation), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "net_output_translation_confidence: {}".format(net_output_translation_confidence), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "{}".format(net_output_translation), (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 5, cv2.LINE_AA)

    # print("Displaying cropped frame...")
    # # 显示摄像头窗口
    # cv2.imshow('How you are doing', cropped_frame)

    # # resize camera window
    # # camera_frame_resized = cv2.resize(original_frame, (camera_width, camera_height))
    # # # 显示摄像头画面窗口，设置为屏幕右边
    # cv2.imshow('How you are doing', camera_frame_resized)
    # cv2.moveWindow('How you are doing', camera_width, 0)  # 右边

    # cv2.namedWindow("Analytics", cv2.WND_PROP_FULLSCREEN)
    # cv2.setWindowProperty("Analytics", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
   
    # 获取屏幕分辨率
    screen_width = 1920  # 屏幕宽度
    screen_height = 1080  # 屏幕高度


    # 创建窗口并设置为全屏
    cv2.namedWindow("Analytics", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Analytics", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    
    # 创建一个小窗口来显示 Hand Segmentation
    cv2.namedWindow("Hand Segmentation", cv2.WND_PROP_FULLSCREEN)
    # 可以设置窗口的位置来确保它显示在 Analytics 窗口之上
    cv2.moveWindow("Hand Segmentation", 100, 100)  # 调整这里的位置

    # 假设你有一个全屏的 `analytics_frame` 和一个小的 `hand_segmentation_frame`
    # analytics_frame = np.zeros((screen_height, screen_width, 3), dtype=np.uint8)
    hand_segmentation_frame = np.zeros((200, 300, 3), dtype=np.uint8)  # 小的手部分割图像

    # 在 Analytics 窗口中显示更新后的图像
    cv2.imshow("Analytics", analytics_frame)

    # 在 Hand Segmentation 窗口中显示更新后的图像
    cv2.imshow("Hand Segmentation", hand_segmentation_frame)

        
        
    if cv2.waitKey(25) == ord('q'):
            break

video_capture.release()
cv2.destroyAllWindows()
