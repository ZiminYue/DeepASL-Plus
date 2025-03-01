import cv2
import mediapipe as mp
import math
import statistics
import torch
import torch.nn as nn
# import net_init
import torch.nn.functional as F
import numpy as np

# 定义你的神经网络架构
class Net(nn.Module):
    def __init__(self):
        super(Net, self).__init__()
        self.conv1 = nn.Conv2d(1, 16, kernel_size=3)
        self.conv2 = nn.Conv2d(16, 32, kernel_size=3)
        self.fc1 = nn.Linear(32 * 5 * 5, 128)
        self.fc2 = nn.Linear(128, 36)

    def forward(self, x):
        x = F.relu(self.conv1(x))
        x = F.max_pool2d(x, 2)
        print(x.shape)

        x = F.relu(self.conv2(x))
        x = F.max_pool2d(x, 2)
        print(x.shape)


        x = x.view(x.size(0), -1)  # 展平操作
        print(x.shape)
        
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return F.log_softmax(x, dim=1)

# 创建模型实例
model = Net()

video_capture = cv2.VideoCapture(0)

#Test if camera is opened (can made to be seen for the users later)
if not video_capture.isOpened():
    print("Cannot access the camera.")
    exit()
else:
    print("Camera OK.")

# 设置分辨率
video_capture.set(cv2.CAP_PROP_FRAME_WIDTH, 1920)
video_capture.set(cv2.CAP_PROP_FRAME_HEIGHT, 1080)

#Initiate mediapipe and openCV
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
    alphabet_and_digits = ['0','1','2','3','4','5','6','7','8','9','A', 'B', 'C', 'D', 'E', 'F', 'G', 'H', 'I', 'J', 'K', 'L', 'M', 'N', 'O', 'P', 'Q', 'R', 'S', 'T', 'U', 'V', 'W', 'X', 'Y']
    translation = alphabet_and_digits[softmax_argmax]

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
            if foward_threshold is not None:
                forward_inverse_threshold = cv2.threshold(foward_threshold, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
                forward_inverse_threshold = cv2.flip(forward_inverse_threshold, 1)
            else:
                forward_inverse_threshold = np.zeros_like(forward_frame)  # If no valid foward_threshold, show a blank screen

            forward_inverse_threshold = cv2.threshold(foward_threshold, 0, 255, cv2.THRESH_BINARY_INV + cv2.THRESH_OTSU)[1] 
            forward_inverse_threshold = cv2.flip(forward_inverse_threshold, 1)

            
            cv2.imshow('Your Gesture', forward_inverse_threshold)
            cv2.moveWindow("Your Gesture", 100, 700)  # adjust gesture window position  
            
        

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
        
        
                # net_softmax_output = net_init.trained_forward_propagation(forward_inverse_threshold_resized)
                # 确保输入数据是一个 4D Tensor
                if len(forward_inverse_threshold_resized.shape) == 2:  # 2D -> 28x28 图像
                    forward_inverse_threshold_resized = torch.tensor(forward_inverse_threshold_resized).unsqueeze(0)  # 增加批次维度 [1, 28, 28]
                    forward_inverse_threshold_resized = forward_inverse_threshold_resized.unsqueeze(0)  # 增加通道维度 [1, 1, 28, 28] (单通道)

                # 如果是多张图像，形状为 [batch_size, 28, 28]，可以直接转换为 [batch_size, 1, 28, 28]
                elif len(forward_inverse_threshold_resized.shape) == 3:  # 多张图像
                    forward_inverse_threshold_resized = torch.tensor(forward_inverse_threshold_resized).unsqueeze(1)  # [batch_size, 1, 28, 28]
                    
                # 确保是 float 类型
                forward_inverse_threshold_resized = forward_inverse_threshold_resized.float()
            
                net_softmax_output = model(forward_inverse_threshold_resized)
                # net_softmax_output_list = net_softmax_output.tolist()
               # 转换为概率值
                net_softmax_output_list = F.softmax(net_softmax_output, dim=1).detach().cpu().numpy().flatten()

                print("Softmax output list: ", net_softmax_output_list)  # 打印输出列表

                # 确保列表非空并且有正确的元素
                if len(net_softmax_output_list) > 0:
                    # 取出最大概率的索引
                    predicted_class = np.argmax(net_softmax_output_list)
                    print(f"Predicted class: {predicted_class} with confidence: {net_softmax_output_list[predicted_class]:.2%}")
                else:
                    print("The softmax output list is empty!")

                # 假设 softmax_translation 是一个将索引转换为类别名称的函数
                net_output_translation = str(softmax_translation(predicted_class))
                print("Translated output: ", net_output_translation)

                net_output_translation_confidence = "{:.2%}".format(net_softmax_output_list[np.argmax(net_softmax_output_list)])
                # net_output_translation = str(softmax_translation(np.argmax(net_softmax_output)))              
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


    cv2.putText(analytics_frame, "[Press ESC to exit]", (10, 20), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 98, 255), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "hand_in_frame: {}".format(str(hand_in_frame)), (10, 50), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "hand_position_eval: {}".format(hand_position_eval), (10, 80), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "anchor_distances_avg: {}".format(float(anchor_distances_avg)), (10, 110), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "hand_centroid_velocity: {}".format(hand_centroid_velocity), (10, 140), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "net_testing: {}".format(net_testing), (10, 170), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "net_output_translation: {}".format(net_output_translation), (10, 200), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "net_output_translation_confidence: {}".format(net_output_translation_confidence), (10, 230), cv2.FONT_HERSHEY_SIMPLEX, 0.75, (0, 255, 0), 2, cv2.LINE_AA)
    cv2.putText(analytics_frame, "{}".format(net_output_translation), (10, 600), cv2.FONT_HERSHEY_SIMPLEX, 10, (0, 0, 255), 5, cv2.LINE_AA)


   
    # 获取屏幕分辨率
    screen_width = 1920  # 屏幕宽度
    screen_height = 1080  # 屏幕高度


    # 创建窗口并设置为全屏
    cv2.namedWindow("Analytics", cv2.WND_PROP_FULLSCREEN)
    cv2.setWindowProperty("Analytics", cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)
    

    # 在 Analytics 窗口中显示更新后的图像
    cv2.imshow("Analytics", analytics_frame)
        
        
    if cv2.waitKey(25) == 27:
            break

video_capture.release()
cv2.destroyAllWindows()
