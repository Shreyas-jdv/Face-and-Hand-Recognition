import cv2
import time
import mediapipe as mp

mp_holistic = mp.solutions.holistic
holistic_model = mp_holistic.Holistic(
    min_detection_confidence=0.5,
    min_tracking_confidence=0.5
)

mp_drawing = mp.solutions.drawing_utils

capture = cv2.VideoCapture(0)

def count_extended_fingers(hand_landmarks):
    finger_states = [0] * 5  # Initialize finger states as closed
    # Define finger landmark indices
    finger_tip_indices = [8, 12, 16, 20,]  # Exclude thumb
    for idx, tip_index in enumerate(finger_tip_indices):
        tip_landmark = hand_landmarks.landmark[tip_index]
        # Check if the tip landmark is above the corresponding pip and dip landmarks
        if tip_landmark.y < hand_landmarks.landmark[tip_index - 2].y and \
           tip_landmark.y < hand_landmarks.landmark[tip_index - 3].y:
            finger_states[idx + 1] = 1  # Finger is extended
    return finger_states

previousTime = 0
currentTime = 0

while capture.isOpened():
    ret, frame = capture.read()
    frame = cv2.resize(frame, (800, 600))
    image = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

    image.flags.writeable = False
    results = holistic_model.process(image)
    image.flags.writeable = True

    image = cv2.cvtColor(image, cv2.COLOR_RGB2BGR)

    if results.left_hand_landmarks:
        left_finger_states = count_extended_fingers(results.left_hand_landmarks)
        left_hand_extended_fingers = sum(left_finger_states)
        cv2.putText(image, f"Left Hand Fingers: {left_hand_extended_fingers}", (10, 100),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    if results.right_hand_landmarks:
        right_finger_states = count_extended_fingers(results.right_hand_landmarks)
        right_hand_extended_fingers = sum(right_finger_states)
        cv2.putText(image, f"Right Hand Fingers: {right_hand_extended_fingers}", (10, 130),
                    cv2.FONT_HERSHEY_COMPLEX, 1, (255, 255, 0), 2)

    mp_drawing.draw_landmarks(
        image,
        results.face_landmarks,
        mp_holistic.FACEMESH_CONTOURS,
        mp_drawing.DrawingSpec(color=(255, 0, 255), thickness=1, circle_radius=1),
        mp_drawing.DrawingSpec(color=(0, 255, 255), thickness=1, circle_radius=1)
    )

    mp_drawing.draw_landmarks(
        image,
        results.right_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    mp_drawing.draw_landmarks(
        image,
        results.left_hand_landmarks,
        mp_holistic.HAND_CONNECTIONS
    )

    currentTime = time.time()
    fps = 1 / (currentTime-previousTime)
    previousTime = currentTime

    cv2.putText(image, str(int(fps))+" FPS", (10, 70), cv2.FONT_HERSHEY_COMPLEX, 1, (0, 255, 0), 2)

    cv2.imshow("Facial and Hand Landmarks", image)

    if cv2.waitKey(5) & 0xFF == ord('q'):
        break

capture.release()
cv2.destroyAllWindows()