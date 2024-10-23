from flask import Flask, render_template, Response
import cv2
import mediapipe as mp
import numpy as np

app = Flask(__name__)

# Initialize MediaPipe and OpenCV (from your original code)
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

write_mode = False
erase_mode = False
stop_mode = True
previous_point = None
canvas = None

def generate_frames():
    global write_mode, erase_mode, stop_mode, previous_point, canvas

    cap = cv2.VideoCapture(0)
    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        h, w, _ = img.shape
        
        if canvas is None:
            canvas = np.zeros((h, w, 3), dtype=np.uint8)

        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                mp_draw.draw_landmarks(img, hand_landmarks, mp_hands.HAND_CONNECTIONS)

                thumb_tip = hand_landmarks.landmark[mp_hands.HandLandmark.THUMB_TIP]
                index_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.INDEX_FINGER_TIP]
                middle_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.MIDDLE_FINGER_TIP]
                ring_finger_tip = hand_landmarks.landmark[mp_hands.HandLandmark.RING_FINGER_TIP]
                pinky_tip = hand_landmarks.landmark[mp_hands.HandLandmark.PINKY_TIP]

                cx_thumb, cy_thumb = int(thumb_tip.x * w), int(thumb_tip.y * h)
                cx_index, cy_index = int(index_finger_tip.x * w), int(index_finger_tip.y * h)
                cx_middle, cy_middle = int(middle_finger_tip.x * w), int(middle_finger_tip.y * h)
                cx_ring, cy_ring = int(ring_finger_tip.x * w), int(ring_finger_tip.y * h)
                cx_pinky, cy_pinky = int(pinky_tip.x * w), int(pinky_tip.y * h)

                index_middle_dist = np.sqrt((cx_index - cx_middle) ** 2 + (cy_index - cy_middle) ** 2)
                middle_ring_dist = np.sqrt((cx_middle - cx_ring) ** 2 + (cy_middle - cy_ring) ** 2)
                ring_pinky_dist = np.sqrt((cx_ring - cx_pinky) ** 2 + (cy_ring - cy_pinky) ** 2)

                distance_threshold = 30

                if np.sqrt((cx_thumb - cx_index) ** 2 + (cy_thumb - cy_index) ** 2) < 40:
                    write_mode = True
                    erase_mode = False
                    stop_mode = False
                elif (index_middle_dist < distance_threshold and 
                      middle_ring_dist < distance_threshold and 
                      ring_pinky_dist < distance_threshold):
                    write_mode = False
                    erase_mode = True
                    stop_mode = False
                else:
                    write_mode = False
                    erase_mode = False
                    stop_mode = True

                if write_mode:
                    if previous_point:
                        cv2.line(canvas, previous_point, (cx_index, cy_index), (0, 0, 255), thickness=5)
                    previous_point = (cx_index, cy_index)
                elif erase_mode:
                    cv2.circle(canvas, (cx_index, cy_index), 50, (0, 0, 0), -1)
                    previous_point = (cx_index, cy_index)
                else:
                    previous_point = None

        img_combined = cv2.addWeighted(img, 0.5, canvas, 0.5, 0)
        ret, buffer = cv2.imencode('.jpg', img_combined)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

if __name__ == "__main__":
    app.run(debug=True)
