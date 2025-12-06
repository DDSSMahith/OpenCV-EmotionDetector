import cv2
import mediapipe as mp
import numpy as np

mp_face_mesh = mp.solutions.face_mesh
mp_draw = mp.solutions.drawing_utils


def get_point(lm, w, h):
    return int(lm.x * w), int(lm.y * h)


def dist(a, b):
    return np.linalg.norm(np.subtract(a, b))


# ----------------------------------------
# EMOTION CLASSIFIER (Rule-Based)
# ----------------------------------------
def predict_emotion(landmarks, w, h):
    # Key points
    top_lip = get_point(landmarks[13], w, h)
    bottom_lip = get_point(landmarks[14], w, h)

    left_eye_up = get_point(landmarks[159], w, h)
    left_eye_down = get_point(landmarks[145], w, h)

    right_eye_up = get_point(landmarks[386], w, h)
    right_eye_down = get_point(landmarks[374], w, h)

    left_brow = get_point(landmarks[70], w, h)
    right_brow = get_point(landmarks[300], w, h)

    nose = get_point(landmarks[1], w, h)

    left_mouth_corner = get_point(landmarks[61], w, h)
    right_mouth_corner = get_point(landmarks[291], w, h)

    # Measurements
    mouth_open = dist(top_lip, bottom_lip)
    eye_open_left = dist(left_eye_up, left_eye_down)
    eye_open_right = dist(right_eye_up, right_eye_down)
    brow_raise_left = dist(left_brow, left_eye_up)
    brow_raise_right = dist(right_brow, right_eye_up)
    mouth_asym = abs(left_mouth_corner[1] - right_mouth_corner[1])

    # Normalize relative to face width
    face_width = dist(left_mouth_corner, right_mouth_corner)
    mouth_open /= face_width
    eye_open_left /= face_width
    eye_open_right /= face_width
    brow_raise_left /= face_width
    brow_raise_right /= face_width
    mouth_asym /= face_width

    # ----------------------
    # Emotion logic
    # ----------------------

    # Surprise
    if mouth_open > 0.28 and eye_open_left > 0.12 and eye_open_right > 0.12:
        return "Surprised 😮"

    # Happy
    if mouth_open > 0.18 and eye_open_left < 0.09:
        return "Happy 🙂"

    # Sad
    if brow_raise_left < 0.04 and brow_raise_right < 0.04:
        return "Sad 🙁"

    # Angry
    if brow_raise_left < 0.045 and brow_raise_right < 0.045 and mouth_open < 0.12:
        return "Angry 😠"

    # Disgust
    if (top_lip[1] - bottom_lip[1]) < 0 and mouth_open < 0.1:
        return "Disgust 😒"

    # Fear
    if mouth_open > 0.15 and (brow_raise_left > 0.09 or brow_raise_right > 0.09):
        return "Fear 😨"

    # Contempt
    if mouth_asym > 0.08:
        return "Contempt 😏"

    # Confused
    if brow_raise_left > 0.1 and brow_raise_right < 0.05:
        return "Confused 🤨"

    # Neutral
    return "Neutral 😐"


# ----------------------------------------
# MAIN PROGRAM
# ----------------------------------------
def main():
    cap = cv2.VideoCapture(0)

    with mp_face_mesh.FaceMesh(
        max_num_faces=1,
        refine_landmarks=True,
        min_detection_confidence=0.5,
        min_tracking_confidence=0.5
    ) as face_mesh:

        while True:
            ret, frame = cap.read()
            if not ret:
                break

            h, w, c = frame.shape
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
            results = face_mesh.process(rgb)

            if results.multi_face_landmarks:
                for face_landmarks in results.multi_face_landmarks:

                    mp_draw.draw_landmarks(
                        frame,
                        face_landmarks,
                        mp_face_mesh.FACEMESH_TESSELATION,
                        mp_draw.DrawingSpec(color=(0, 255, 0), thickness=1, circle_radius=1),
                        mp_draw.DrawingSpec(color=(255, 0, 0), thickness=1)
                    )

                    emotion = predict_emotion(face_landmarks.landmark, w, h)

                    cv2.putText(frame, emotion, (30, 50),
                                cv2.FONT_HERSHEY_SIMPLEX,
                                1.3, (255, 255, 255), 3)

            cv2.imshow("Emotion Detection - Advanced Model", frame)

            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    cap.release()
    cv2.destroyAllWindows()


if __name__ == "__main__":
    main()
