import numpy as np
import cv2
import scipy.io

# Load the .mat file
data = scipy.io.loadmat('/home/darklord/Downloads/RHD_v1-1/RHD_published_v2/training/anno_training.mat')

# Access the desired frame data
frame_key = 'frame43'  # Adjust this according to your frame key
frame_data = data[frame_key]

# Extract 2D keypoints
keypoints_2D = frame_data[0][0][1]  # Assuming this is how to access it based on your structure
keypoints_2D = keypoints_2D[:, :2]  # Get only the x and y coordinates, discard the homogeneous part

# Load your image
image_path = '/home/darklord/Downloads/RHD_v1-1/RHD_published_v2/training/color/00043.png'  # Path to your image
image = cv2.imread(image_path)

# Draw keypoints on the image
for point in keypoints_2D:
    x, y = int(point[0]), int(point[1])
    cv2.circle(image, (x, y), 5, (0, 255, 0), -1)  # Draw a green circle with radius 5 at each keypoint

# Show the image with keypoints
cv2.imshow('Keypoints Image', image)
cv2.waitKey(0)
cv2.destroyAllWindows()