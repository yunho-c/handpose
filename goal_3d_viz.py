import cv2
# import mediapipe as mp
import handpose as hp
import numpy as np

# Initialize MediaPipe Hands.
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
hp_hands = hp.solutions.hands
hp_drawing = hp.solutions.drawing_utils
# hands = mp_hands.Hands(
#     static_image_mode=False,
#     max_num_hands=1,
#     min_detection_confidence=0.5,
#     min_tracking_confidence=0.5)
hands = hp_hands.Hands( # NOTE feel free to change the API
    static_image_mode=False,
    max_num_hands=1,
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5)

# Define colors for each finger.
finger_colors = {
    'thumb': (80, 70, 180),   # Red
    'index': (0, 255, 0),     # Green
    'middle': (255, 0, 0),    # Blue
    'ring': (0, 255, 255),    # Cyan
    'pinky': (255, 255, 0)    # Yellow
}

# Finger connections.
finger_connections = {
    'thumb': [1, 2, 3, 4],
    'index': [5, 6, 7, 8],
    'middle': [9, 10, 11, 12],
    'ring': [13, 14, 15, 16],
    'pinky': [17, 18, 19, 20]
}

# Start capturing video input from the camera.
cap = cv2.VideoCapture(0)

while cap.isOpened():
    success, image = cap.read()
    if not success:
        print("Ignoring empty camera frame.")
        continue

    # Flip the image horizontally for a later selfie-view display, and convert the BGR image to RGB.
    image = cv2.cvtColor(cv2.flip(image, 1), cv2.COLOR_BGR2RGB)

    # Process the image and find hands.
    results = hands.process(image)

    # Convert the image color back so it can be displayed.
    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    # Draw the hand annotations on the image.
    # if results.multi_hand_landmarks:
    if results['multi_hand_landmarks']:
        # for hand_landmarks in results.multi_hand_landmarks:
        for hand_landmarks in results['multi_hand_landmarks']:
            for finger_name, finger_idxs in finger_connections.items():
                for idx in range(len(finger_idxs) - 1):
                    start_idx = finger_idxs[idx]
                    end_idx = finger_idxs[idx + 1]
                    cv2.line(image, 
                            #  (int(hand_landmarks.landmark[start_idx].x * image.shape[1]), 
                            #   int(hand_landmarks.landmark[start_idx].y * image.shape[0])),
                            #  (int(hand_landmarks.landmark[end_idx].x * image.shape[1]), 
                            #   int(hand_landmarks.landmark[end_idx].y * image.shape[0])),
                             (int(hand_landmarks[start_idx][0] * image.shape[1]), 
                              int(hand_landmarks[start_idx][1] * image.shape[0])),
                             (int(hand_landmarks[end_idx][0] * image.shape[1]), 
                              int(hand_landmarks[end_idx][1] * image.shape[0])),
                             finger_colors[finger_name], 2)

    # Display the resulting image.
    # cv2.imshow('MediaPipe Hands', image)
    cv2.imshow('HandPose-Python Hands', image)

    # Break the loop when 'q' is pressed.
    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

hands.close()
cap.release()
