from flask import Flask, render_template, Response, request
import cv2
import mediapipe as mp
import numpy as np
from flask import send_file
import io
import pytesseract
from sympy import sympify

app = Flask(__name__)

# Initialize MediaPipe and OpenCV
mp_hands = mp.solutions.hands
hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)
mp_draw = mp.solutions.drawing_utils

write_mode = False
erase_mode = False
stop_mode = True
math_mode = False
previous_point = None
canvas = None
current_color = (0, 0, 0)
current_thickness = 5
def generate_frames():
    global write_mode, erase_mode, stop_mode, math_mode, previous_point, canvas, current_color, current_thickness

    cap = cv2.VideoCapture(0)
    success, img = cap.read()
    if success:
        h, w, _ = img.shape
        if canvas is None:
            # Initialize white canvas
            canvas = np.ones((h, w, 3), dtype=np.uint8) * 255

    while True:
        success, img = cap.read()
        if not success:
            break

        img = cv2.flip(img, 1)
        img_rgb = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
        results = hands.process(img_rgb)

        # Create a copy of the canvas for displaying hand landmarks
        display_frame = canvas.copy()

        if results.multi_hand_landmarks:
            for hand_landmarks in results.multi_hand_landmarks:
                # Draw hand landmarks on the display frame
                mp_draw.draw_landmarks(display_frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

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
                        cv2.line(canvas, previous_point, (cx_index, cy_index), current_color, current_thickness)
                    previous_point = (cx_index, cy_index)
                elif erase_mode:
                    cv2.circle(canvas, (cx_index, cy_index), 50, (255, 255, 255), -1)
                    previous_point = (cx_index, cy_index)
                else:
                    previous_point = None

        ret, buffer = cv2.imencode('.jpg', display_frame)
        frame = buffer.tobytes()
        
        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/save_image')
def save_image():
    global canvas
    if canvas is not None:
        # Convert the canvas to an image
        _, buffer = cv2.imencode('.png', canvas)
        io_buf = io.BytesIO(buffer)
        io_buf.seek(0)
        return send_file(io_buf, mimetype='image/png', as_attachment=True, download_name='canvas.png')
    return 'No canvas to save', 400

from fpdf import FPDF

@app.route('/save_pdf')
def save_pdf():
    global canvas
    if canvas is not None:
        # Convert the canvas to an image
        success, buffer = cv2.imencode('.png', canvas)
        if success:
            # Save the image to a temporary file
            temp_image_path = 'temp_image.png'
            with open(temp_image_path, 'wb') as f:
                f.write(buffer)

            # Create a PDF and add the image
            pdf = FPDF()
            pdf.add_page()
            pdf.image(temp_image_path, x=10, y=10, w=190)  # Adjust the position and size as needed

            # Save the PDF to a temporary file
            temp_pdf_path = 'canvas.pdf'
            pdf.output(temp_pdf_path)

            # Send the PDF file as a response
            return send_file(temp_pdf_path, mimetype='application/pdf', as_attachment=True, download_name='canvas.pdf')
    return 'No canvas to save', 400

@app.route('/clear')
def clear_canvas():
    global canvas
    if canvas is not None:
        canvas = np.ones_like(canvas) * 255
    return 'Canvas cleared'

@app.route('/set_color')
def set_color():
    global current_color
    color = request.args.get('color', default='#000000')
    
    # Ensure color is in the format #RRGGBB
    if color.startswith('#') and len(color) == 7:
        # Convert RGB to BGR
        current_color = (
            int(color[5:7], 16),  # Blue (OpenCV expects BGR)
            int(color[3:5], 16),  # Green
            int(color[1:3], 16)   # Red
        )
    else:
        # If the color is invalid, set to black (default)
        current_color = (0, 0, 0)

    return 'Color set'


@app.route('/set_thickness')
def set_thickness():
    global current_thickness
    thickness = request.args.get('thickness', default=5, type=int)
    current_thickness = thickness
    return 'Thickness set'

if __name__ == "__main__":
    app.run(debug=True, port=3000)