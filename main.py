# import cv2
# import numpy as np
# import joblib
# import mediapipe as mp
# from flask import Flask, render_template, request, jsonify
# from gtts import gTTS
# import os
# from io import BytesIO
# # Initialize Flask app
# app = Flask(__name__)

# # Load model and label encoder
# model_dict = joblib.load('model.pkl')
# model = model_dict['model']
# with open('label_encoder.pkl', 'rb') as f:
#     label_encoder = joblib.load(f)

# # Default dictionary (could be changed based on user selection)

# mp_hands = mp.solutions.hands
# mp_drawing = mp.solutions.drawing_utils
# mp_drawing_styles = mp.solutions.drawing_styles

# hands = mp_hands.Hands(static_image_mode=False, min_detection_confidence=0.7)  # Adjusted confidence

# def process_frame(frame, selected_dict):
#     H, W, _ = frame.shape
#     frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
#     results = hands.process(frame_rgb)
    
#     if results.multi_hand_landmarks:
#         data_aux = [0] * 84  # Initialize with zeros for two hands
#         hand_idx = 0  # Index to track the hand being processed

#         for hand_landmarks in results.multi_hand_landmarks:
#             if hand_idx >= 2:  # Skip if more than two hands are detected
#                 break

#             x_ = []
#             y_ = []

#             for i, landmark in enumerate(hand_landmarks.landmark):
#                 x = landmark.x
#                 y = landmark.y

#                 x_.append(x)
#                 y_.append(y)

#             min_x, max_x = min(x_), max(x_)
#             min_y, max_y = min(y_), max(y_)

#             normalized = []
#             for x, y in zip(x_, y_):
#                 normalized.append((x - min_x) / (max_x - min_x))
#                 normalized.append((y - min_y) / (max_y - min_y))

#             data_aux[hand_idx * 42:(hand_idx + 1) * 42] = normalized
#             hand_idx += 1

#         prediction = model.predict([np.asarray(data_aux)])
#         predicted_label = label_encoder.inverse_transform(prediction)[0]
#         print(f"Predicted Label: {predicted_label}")

#         # Translate the prediction using the selected dictionary
#         predicted_character = selected_dict.get(predicted_label, "Unknown")
#         print(f"Predicted Character: {predicted_character}")  # Debugging line

#         # Draw bounding boxes and predictions on the frame
#         for hand_landmarks in results.multi_hand_landmarks:
#             mp_drawing.draw_landmarks(
#                 frame, 
#                 hand_landmarks, 
#                 mp_hands.HAND_CONNECTIONS, 
#                 mp_drawing_styles.get_default_hand_landmarks_style(),
#                 mp_drawing_styles.get_default_hand_connections_style()
#             )

#         return predicted_character, frame
#     else:
#         return None, frame


# # @app.route('/')
# # def index():
# #     return render_template('index.html')


# @app.route('/predict', methods=['POST'])
# def predict():
#     try:
#         if 'image' not in request.files:
#             return jsonify({'error': 'No image provided'}), 400

#         # Get the selected language
#         lang = request.args.get('lang', 'en')  # Default to English if no lang parameter is provided
#         print(f"Received language: {lang}")  # Debugging line
        
#         # Select dictionary based on language
#         language_dict = {
#             'en': english_dict,
#             'hi': hindi_dict,
#             'bn': bengali_dict,
#             'ml': malayalam_dict,
#             'mr': marathi_dict,
#             'pa': punjabi_dict,
#             'ta': tamil_dict,
#             'te': telugu_dict,
#             'kn': kannada_dict,
#             'gu': gujarati_dict,
#             'ur': urdu_dict
#         }

#         selected_dict = language_dict.get(lang, english_dict)  # Fallback to English if language not found

#         print(f"Selected dictionary: {selected_dict}")  # Debugging line
        
#         file = request.files['image'].read()
#         np_img = np.frombuffer(file, np.uint8)
#         frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)

#         prediction, processed_frame = process_frame(frame, selected_dict)

#         if prediction is None:
#             return jsonify({'error': 'No hand landmarks detected or feature length mismatch'}), 400

#         # Encode the processed frame back to JPEG format
#         _, buffer = cv2.imencode('.jpg', processed_frame)
#         frame_data = buffer.tobytes()

#         return jsonify({
#             'prediction': prediction,  # Return the full prediction text
#             'image': frame_data.hex()
#         })

#     except Exception as e:
#         return jsonify({'error': str(e)}), 500

# @app.route('/tts', methods=['POST'])
# def text_to_speech():
#     text = request.json.get('text')
#     lang = request.json.get('lang', 'en')  # Default to English if no language provided

#     if not text:
#         return {'error': 'No text provided'}, 400

#     try:
#         # Convert text to speech
#         tts = gTTS(text=text, lang=lang)
#         audio_file = BytesIO()
#         tts.write_to_fp(audio_file)
#         audio_file.seek(0)
        
#         # Send the audio file back as a response
#         return send_file(audio_file, mimetype='audio/mp3', as_attachment=True, download_name='speech.mp3')
#     except Exception as e:
#         return {'error': str(e)}, 500
# if __name__ == '__main__':
#     app.run(host='0.0.0.0', port=5000)
import cv2
import numpy as np
import joblib
import base64
from flask import Flask, request, jsonify
from io import BytesIO
from flask_cors import CORS
app = Flask(__name__)
CORS(app)

# Load model & label encoder
model_dict = joblib.load('model.pkl')
model = model_dict['model']
label_encoder = joblib.load('label_encoder.pkl')

# MediaPipe hands setup
import mediapipe as mp
mp_hands = mp.solutions.hands
mp_drawing = mp.solutions.drawing_utils
mp_styles = mp.solutions.drawing_styles
hands = mp_hands.Hands(
    static_image_mode=False,
    max_num_hands=2,
    min_detection_confidence=0.7,
    min_tracking_confidence=0.7
)

def process_frame(frame):
    """Detect hands, extract normalized 84‑length feature vector, predict label."""
    rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)
    results = hands.process(rgb)

    # prepare zero vector for up to 2 hands
    features = np.zeros(84, dtype=np.float32)

    if results.multi_hand_landmarks:
        for idx, hand in enumerate(results.multi_hand_landmarks):
            if idx >= 2: break
            xs = [lm.x for lm in hand.landmark]
            ys = [lm.y for lm in hand.landmark]
            min_x, max_x = min(xs), max(xs)
            min_y, max_y = min(ys), max(ys)
            w = max_x - min_x or 1.0
            h = max_y - min_y or 1.0

            norm = []
            for x, y in zip(xs, ys):
                norm.append((x - min_x) / w)
                norm.append((y - min_y) / h)

            features[idx*42:(idx+1)*42] = norm

        pred_idx = model.predict(features.reshape(1, -1))[0]
        label = label_encoder.inverse_transform([pred_idx])[0]
    else:
        label = None

    # draw landmarks on frame
    if results.multi_hand_landmarks:
        for hand in results.multi_hand_landmarks:
            mp_drawing.draw_landmarks(
                frame, hand, mp_hands.HAND_CONNECTIONS,
                mp_styles.get_default_hand_landmarks_style(),
                mp_styles.get_default_hand_connections_style()
            )

    return label, frame

@app.route('/predict', methods=['POST'])
def predict():
    if 'image' not in request.files:
        return jsonify({'error': 'No image provided'}), 400

    # read image bytes into OpenCV format
    file_bytes = request.files['image'].read()
    np_img = np.frombuffer(file_bytes, np.uint8)
    frame = cv2.imdecode(np_img, cv2.IMREAD_COLOR)
    if frame is None:
        return jsonify({'error': 'Invalid image'}), 400

    label, proc = process_frame(frame)
    if label is None:
        return jsonify({'error': 'No hands detected'}), 200

    # encode processed frame to JPEG then base64
    _, buf = cv2.imencode('.jpg', proc)
    jpg_b64 = base64.b64encode(buf).decode('utf-8')

    return jsonify({
        'prediction': label,
        'image': jpg_b64
    })

if __name__ == '__main__':
    app.run(debug=True)
