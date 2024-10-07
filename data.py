import scipy.io
import os
data = scipy.io.loadmat('/home/darklord/Downloads/RHD_v1-1/RHD_published_v2/training/anno_training.mat')

data_dir = '/home/darklord/datasets/2d_hand_dataset/training'
os.makedirs(data_dir, exist_ok=True)
for key in data.keys():
    if key.startswith('frame'):
        filename = key[5:].zfill(5) + '.txt'
        file_path = os.path.join(data_dir, filename)

        if not os.path.exists(file_path):
            keypoints = data[key][0][0][1]
            
         
            if len(keypoints) < 42: 
                print(f"Not enough keypoints for {key}. Skipping.")
                continue
            x_min_left, y_min_left = float('inf'), float('inf')
            x_max_left, y_max_left = 0, 0
            for keypoint in keypoints[:21]:
                x_left, y_left, z_left = keypoint
                x_min_left = min(x_min_left, x_left)
                y_min_left = min(y_min_left, y_left)
                x_max_left = max(x_max_left, x_left)
                y_max_left = max(y_max_left, y_left)

            width_left = x_max_left - x_min_left
            height_left = y_max_left - y_min_left
            x_min_right, y_min_right = float('inf'), float('inf')
            x_max_right, y_max_right = 0, 0
            for keypoint in keypoints[21:]:
                x_right, y_right, z_right = keypoint
                x_min_right = min(x_min_right, x_right)
                y_min_right = min(y_min_right, y_right)
                x_max_right = max(x_max_right, x_right)
                y_max_right = max(y_max_right, y_right)
            width_right = x_max_right - x_min_right
            height_right = y_max_right - y_min_right
            with open(file_path, 'w') as file:
                file.write(f"0 {x_min_left} {y_min_left} {width_left} {height_left} ")
                for keypoint in keypoints[:21]:
                    file.write(" ".join(map(str, keypoint)) + " ")
                file.write(f"\n1 {x_min_right} {y_min_right} {width_right} {height_right} ")
                for keypoint in keypoints[21:]:
                    file.write(" ".join(map(str, keypoint)) + " ")


