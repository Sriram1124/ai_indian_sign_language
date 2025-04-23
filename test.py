import cv2
import numpy as np
import joblib
import mediapipe as mp

def main():
    # ─── Load model & label encoder ───────────────────────────────────────────
    model_dict = joblib.load('model.pkl')
    model = model_dict['model']
    label_encoder = joblib.load('label_encoder.pkl')

    # ─── MediaPipe Hands Setup ────────────────────────────────────────────────
    mp_hands = mp.solutions.hands
    mp_drawing = mp.solutions.drawing_utils
    mp_styles = mp.solutions.drawing_styles

    hands = mp_hands.Hands(
        static_image_mode=False,
        max_num_hands=2,
        min_detection_confidence=0.7,
        min_tracking_confidence=0.7
    )

    # ─── Open Webcam ───────────────────────────────────────────────────────────
    cap = cv2.VideoCapture(0)
    if not cap.isOpened():
        print("❌ Error: Cannot open webcam.")
        return

    try:
        while True:
            ret, frame = cap.read()
            if not ret:
                print("❌ Failed to grab frame.")
                break

            # Flip & convert color
            frame = cv2.flip(frame, 1)
            rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

            # Detect hands
            results = hands.process(rgb)

            # Prepare a zero‑filled feature vector for up to 2 hands (2×21×2 = 84)
            features = np.zeros(84, dtype=np.float32)

            if results.multi_hand_landmarks:
                for idx, hand_landmarks in enumerate(results.multi_hand_landmarks):
                    if idx >= 2:
                        break

                    xs = [lm.x for lm in hand_landmarks.landmark]
                    ys = [lm.y for lm in hand_landmarks.landmark]

                    min_x, max_x = min(xs), max(xs)
                    min_y, max_y = min(ys), max(ys)
                    width = max_x - min_x or 1.0
                    height = max_y - min_y or 1.0

                    normalized = []
                    for x, y in zip(xs, ys):
                        normalized.append((x - min_x) / width)
                        normalized.append((y - min_y) / height)

                    features[idx*42:(idx+1)*42] = normalized

                # Run prediction
                pred_idx = model.predict(features.reshape(1, -1))[0]
                label = label_encoder.inverse_transform([pred_idx])[0]
            else:
                label = "No Hands"

            # Draw landmarks
            if results.multi_hand_landmarks:
                for hand_landmarks in results.multi_hand_landmarks:
                    mp_drawing.draw_landmarks(
                        frame,
                        hand_landmarks,
                        mp_hands.HAND_CONNECTIONS,
                        mp_styles.get_default_hand_landmarks_style(),
                        mp_styles.get_default_hand_connections_style()
                    )

            # Overlay prediction
            cv2.putText(
                frame,
                f"Prediction: {label}",
                (10, 30),
                cv2.FONT_HERSHEY_SIMPLEX,
                1,
                (0, 255, 0),
                2,
                cv2.LINE_AA
            )

            cv2.imshow("Real-Time Hand Gesture Recognition", frame)
            if cv2.waitKey(1) & 0xFF == ord('q'):
                break

    finally:
        cap.release()
        cv2.destroyAllWindows()

if __name__ == "__main__":
    main()
