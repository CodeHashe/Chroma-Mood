from flask import Flask, render_template, Response, jsonify
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import time

app = Flask(__name__)

# Load model
model = load_model('emotionsvideomodel.h5')

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
emotion_scores = np.zeros(len(emotion_labels))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades +
                                     'haarcascade_frontalface_default.xml')

cap = cv2.VideoCapture(0)

start_time = None
processing = False


def generate_frames():
    global start_time, processing, emotion_scores

    emotion_scores = np.zeros(len(emotion_labels))
    start_time = time.time()
    processing = True

    while processing:
        if time.time() - start_time >= 10:
            processing = False
            break

        success, frame = cap.read()
        if not success:
            break

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        for (x, y, w, h) in faces:
            cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)

            roi_gray = gray[y:y + h, x:x + w]
            roi_gray = cv2.resize(roi_gray, (48, 48))
            roi_gray = roi_gray.astype('float') / 255.0
            roi_gray = np.expand_dims(roi_gray, axis=0)
            roi_gray = np.expand_dims(roi_gray, axis=-1)

            prediction = model.predict(roi_gray)
            emotion_scores += prediction[0]

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')


@app.route('/')
def index():
    return """
        <h1>Emotion Detector</h1>
        <p>Live webcam feed starts automatically for 10 seconds.</p>
        <img src="/video_feed" width="600">
        <p>After 10 seconds, visit <a href='/result'>/result</a> to see the detected emotion.</p>
    """


@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(),
                    mimetype='multipart/x-mixed-replace; boundary=frame')


@app.route('/result')
def result():
    if np.sum(emotion_scores) == 0:
        return jsonify({"error": "No faces detected."})

    final_index = np.argmax(emotion_scores)
    dominant_emotion = emotion_labels[final_index]

    # Adjustments
    if dominant_emotion == 'Surprise':
        dominant_emotion = 'Happy'
    elif dominant_emotion == 'Disgust':
        dominant_emotion = 'Angry'

    return jsonify({
        "dominant_emotion": dominant_emotion,
        "score": float(emotion_scores[final_index])
    })


if __name__ == "__main__":
    app.run(debug=True)
