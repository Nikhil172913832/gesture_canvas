import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Drawing variables
drawing_mode = False  # Initially not in drawing mode
erase_mode = False    # Erase mode
previous_point = None  # Store the previous point to draw lines smoothly
canvas = None  # Create a canvas to draw on

# Capture from webcam
cap = cv2.VideoCapture(0)

while True:
    success, img = cap.read()
    if not success:
        break

    # Flip image horizontally for natural hand movement
    img = cv2.flip(img, 1)
    h, w, _ = img.shape
    
    # Create a black canvas if it does not exist
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Convert the image to RGB since MediaPipe processes RGB images
    img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # Process the frame for hand landmarks
    results = hands.process(img_rgb)

    # If hand landmarks are detected, process them
    if results.multi_hand_landmarks:
        for hand_landmarks in results.multi_hand_landmarks:
            # Draw hand landmarks on the image for visualization
            mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)
            
            # Get positions of the fingers
            thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
            index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
            middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
            ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
            pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]
            
            # Convert the landmark positions to pixel coordinates
            cx_thumb, cy_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
            cx_index, cy_index = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
            cx_middle, cy_middle = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
            cx_ring, cy_ring = int(ring_finger_tip.x * w), int(ring_finger_tip.y * h)
            cx_pinky, cy_pinky = int(pinky_tip.x * w), int(pinky_tip.y * h)

            # Check distance between thumb and index finger tips (for drawing mode)
            thumb_index_dist = np.sqrt((cx_thumb - cx_index) ** 2 + (cy_thumb - cy_index) ** 2)
            
            # Measure the spread of all fingers (for erasing)
            finger_spread = np.sqrt((cx_index - cx_middle)**2 + (cy_index - cy_middle)**2) + \
                            np.sqrt((cx_middle - cx_ring)**2 + (cy_middle - cy_ring)**2) + \
                            np.sqrt((cx_ring - cx_pinky)**2 + (cy_ring - cy_pinky)**2)

            # If all fingers are spread wide enough, enter erase mode (open palm gesture)
            if finger_spread > 250:  # Adjust this threshold if needed
                erase_mode = True
                drawing_mode = False
            else:
                erase_mode = False

            # If the distance is small, activate drawing mode (pinch gesture)
            if not erase_mode and thumb_index_dist < 40:
                drawing_mode = True
            else:
                drawing_mode = False

            # Erase or draw based on the mode
            if erase_mode:
                # Draw a black circle where the index finger tip is to simulate erasing
                cv2.circle(canvas, (cx_index, cy_index), 30, (0, 0, 0), thickness=-1)  # Erase by drawing black circle
            elif drawing_mode:
                if previous_point:
                    # Draw a line from the previous point to the current point on the canvas
                    cv2.line(canvas, previous_point, (cx_index, cy_index), (0, 0, 255), thickness=5)
                previous_point = (cx_index, cy_index)
            else:
                previous_point = None
    
    # Merge the canvas with the webcam feed
    img_combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    # Show the combined image
    cv2.imshow("Gesture Canvas with Erase", img_combined)

    # Exit on 'ESC'
    if cv2.waitKey(1) & 0xFF == 27:
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
