import cv2
from ultralytics import YOLO

# Load your trained YOLO model
model = YOLO('best.pt')  # Replace 'best.pt' with the path to your model

# Open the camera feed
cap = cv2.VideoCapture(0)

while cap.isOpened():
    ret, frame = cap.read()
    if not ret:
        break

    # Run the YOLO model on the frame
    results = model(frame)

    # Visualize the results (bounding boxes, keypoints, etc.)
    annotated_frame = results[0].plot()  # YOLOv8 automatically handles drawing annotations

    # Display the frame with predictions
    cv2.imshow("YOLO Keypoint Detection", annotated_frame)

    # Exit if 'q' is pressed
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

# Release the video capture
cap.release()
cv2.destroyAllWindows()
