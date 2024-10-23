import cv2
import mediapipe as mp
import numpy as np

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

# Drawing variables
write_mode = False  # Initially not in writing mode
erase_mode = False  # Initially not in erase mode
stop_mode = True  # Initially in stop mode
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

            # Check distances between adjacent fingers for erase mode
            index_middle_dist = np.sqrt((cx_index - cx_middle) ** 2 + (cy_index - cy_middle) ** 2)
            middle_ring_dist = np.sqrt((cx_middle - cx_ring) ** 2 + (cy_middle - cy_ring) ** 2)
            ring_pinky_dist = np.sqrt((cx_ring - cx_pinky) ** 2 + (cy_ring - cy_pinky) ** 2)

            # Set a threshold for the distances to detect no gaps
            distance_threshold = 30  # Adjust this for sensitivity

            # If the thumb and index fingers are close together, activate write mode (pinch gesture)
            if np.sqrt((cx_thumb - cx_index) ** 2 + (cy_thumb - cy_index) ** 2) < 40:
                write_mode = True
                erase_mode = False
                stop_mode = False
            # If there are no gaps between the fingers, activate erase mode
            elif (index_middle_dist < distance_threshold and 
                  middle_ring_dist < distance_threshold and 
                  ring_pinky_dist < distance_threshold):
                write_mode = False
                erase_mode = True
                stop_mode = False
            else:
                # Enter stop mode when no gesture is detected
                write_mode = False
                erase_mode = False
                stop_mode = True

            # Write mode: Draw based on finger movement
            if write_mode:
                if previous_point:
                    # Draw a line from the previous point to the current point on the canvas
                    cv2.line(canvas, previous_point, (cx_index, cy_index), (0, 0, 255), thickness=5)
                previous_point = (cx_index, cy_index)
            # Erase mode: Clear the area around the index position
            elif erase_mode:
                # Draw a large filled circle at the index finger's position to erase
                cv2.circle(canvas, (cx_index, cy_index), 50, (0, 0, 0), -1)  # Erase with a black circle
                previous_point = (cx_index, cy_index)  # Update previous point for potential continuity
            else:
                previous_point = None

    # Merge the canvas with the webcam feed
    img_combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)

    # Determine the current mode
    if write_mode:
        mode_text = "Mode: Writing"
    elif erase_mode:
        mode_text = "Mode: Erasing"
    else:
        mode_text = "Mode: Stopped"

    # Add the mode text to the image
    cv2.putText(img_combined, mode_text, (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 1, (255, 255, 255), 2, cv2.LINE_AA)

    # Show the combined image
    cv2.imshow("Gesture Canvas with Write and Erase Mode", img_combined)

    # Exit on 'q'
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the capture and close all windows
cap.release()
cv2.destroyAllWindows()
