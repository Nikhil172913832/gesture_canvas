import cv2
from ultralytics import YOLO
import numpy as np

# Load your trained YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your model

# Drawing variables
write_mode = False  # Initially not in writing mode
stop_mode = True  # Initially in stop mode
previous_point = None  # Store the previous point to draw lines smoothly
canvas = None  # Create a canvas to draw on

# Open the camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    h, w, _ = frame.shape

    # Create a black canvas if it does not exist
    if canvas is None:
        canvas = np.zeros((h, w, 3), dtype=np.uint8)

    # Run the YOLO model on the frame
    results = model(frame)

    # Check if keypoints are detected in the results
    if results[0].keypoints is not None:
        keypoints = results[0].keypoints.xy  # Get keypoints from YOLO results

        # Assuming we are using the first hand only for now (index 0)
        if keypoints.shape[0] > 0:  # Ensure that at least one hand is detected
            keypoints_hand_0 = keypoints[0]  # First hand's keypoints

            # Get specific keypoints for gesture detection
            cx_thumb, cy_thumb = int(keypoints_hand_0[4][0]), int(keypoints_hand_0[4][1])  # Thumb tip (index 4)
            cx_index, cy_index = int(keypoints_hand_0[8][0]), int(keypoints_hand_0[8][1])  # Index finger tip (index 8)

            # Check distance between thumb and index finger tips (for write mode)
            thumb_index_dist = np.sqrt((cx_thumb - cx_index) ** 2 + (cy_thumb - cy_index) ** 2)

            # If the distance is small, activate write mode (pinch gesture)
            if thumb_index_dist < 40:
                write_mode = True
                stop_mode = False
            else:
                # Enter stop mode when no pinch is detected
                write_mode = False
                stop_mode = True

            # Write mode: Draw based on finger movement
            if write_mode:
                if previous_point:
                    # Draw a line from the previous point to the current point on the canvas
                    cv2.line(canvas, previous_point, (cx_index, cy_index), (0, 0, 255), thickness=5)
                previous_point = (cx_index, cy_index)
            else:
                previous_point = None

    # Visualize the results (bounding boxes, keypoints, etc.)
    annotated_frame = results[0].plot()  # YOLOv8 automatically handles drawing annotations

    # Merge the canvas with the webcam feed
    img_combined = cv2.addWeighted(annotated_frame, 0.5, canvas, 0.5, 0)

    # Display the frame with predictions
    cv2.imshow("YOLO Keypoint Detection with Write and Stop Mode", img_combined)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
