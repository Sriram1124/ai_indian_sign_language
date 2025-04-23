# import cv2
# import mediapipe as mp
# import numpy as np
# import pickle

# # Load the trained model
# with open('model.pkl', 'rb') as f:
#     data = pickle.load(f)
# model = data['model']

# # Optional: define your label mapping (update this as per your dataset)
# labels = data.get('labels', {})  # Add a 'labels' key in your pickle if available
# if not labels:
#     labels = {i: f"Label_{i}" for i in range(30)}  # fallback dummy labels

# # Initialize MediaPipe Hands
# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# hands = mp_hands.Hands(static_image_mode=False, max_num_hands=1, min_detection_confidence=0.7)

# # Start webcam
# cap = cv2.VideoCapture(0)

# while cap.isOpened():
#     ret, frame = cap.read()
#     if not ret:
#         break

#     # Flip and convert BGR to RGB
#     frame = cv2.flip(frame, 1)
#     rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

#     # Process hand landmarks
#     result = hands.process(rgb)

#     if result.multi_hand_landmarks:
#         for hand_landmarks in result.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(frame, hand_landmarks, mp_hands.HAND_CONNECTIONS)

#             # Extract landmarks
#             landmark_list = []
#             for lm in hand_landmarks.landmark:
#                 landmark_list.extend([lm.x, lm.y, lm.z])

#             # Check input size
#             if len(landmark_list) == 63:  # Sometimes z is dropped
#                 landmark_list += [0] * (84 - 63)  # Pad if needed
#             if len(landmark_list) == 84:
#                 X = np.array(landmark_list).reshape(1, -1)
#                 prediction = model.predict(X)
#                 label = labels.get(prediction[0], str(prediction[0]))

#                 # Draw prediction
#                 cv2.putText(frame, f'Sign: {label}', (10, 50),
#                             cv2.FONT_HERSHEY_SIMPLEX, 1.5, (0, 255, 0), 3)

#     cv2.imshow("ISL Sign Detection", frame)

#     if cv2.waitKey(1) & 0xFF == 27:  # ESC to quit
#         break

# cap.release()
# cv2.destroyAllWindows()
