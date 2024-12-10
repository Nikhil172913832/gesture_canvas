import cv2
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
def is_finger_up(hand_landmarks, finger_tip_id, finger_mcp_id):
    return hand_landmarks.landmark[finger_tip_id].y < hand_landmarks.landmark[finger_mcp_id].y
FINGER_TIPS = [8, 12, 16, 20]  
FINGER_MCPS = [5, 9, 13, 17] 
def calculate_finger_count(fingers_status):
    if not fingers_status[0]:  
        return 0
    return sum(fingers_status)
cap = cv2.VideoCapture(0)
with mp_hands.Hands(min_detection_confidence=0.5, min_tracking_confidence=0.5) as hands:
    while cap.isOpened():
        ret, frame = cap.read()
        if not ret:
            break
        frame = cv2.flip(frame, 1)
        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
        result = hands.process(frame_rgb)
        if result.multi_hand_landmarks:
            for hand_landmarks in result.multi_hand_landmarks:
                fingers_status = []
                for tip, mcp in zip(FINGER_TIPS, FINGER_MCPS):
                    fingers_status.append(is_finger_up(hand_landmarks, tip, mcp))
                finger_count = calculate_finger_count(fingers_status)
                mp_drawing.draw_landmarks(
                    frame, hand_landmarks, mp_hands.HAND_CONNECTIONS
                )
                cv2.putText(
                    frame, f"Finger Count: {finger_count}",
                    (10, 50), cv2.FONT_HERSHEY_SIMPLEX,
                    1, (0, 255, 0), 2, cv2.LINE_AA
                )
        cv2.imshow("Finger Detection", frame)
        if cv2.waitKey(1) & 0xFF == 27:
            break
cap.release()
cv2.destroyAllWindows()