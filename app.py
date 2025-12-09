import os
import time
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, Response, jsonify
from tensorflow.keras.models import load_model
import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences

app = Flask(__name__)

# --- Configuration ---
VIDEO_MODEL_PATH = os.path.join('VideoEmotions', 'emotionsvideomodel.h5')
VOICE_MODEL_PATH = os.path.join('VoiceModule', 'emotionvoicemodel.keras')
TEXT_MODEL_PATH = os.path.join('Text-to-Emotion', 'Model', 'TexttoEmotion.h5')
VECTORIZER_PATH = os.path.join('Text-to-Emotion', 'Model', 'vectorizer.pkl')
LABEL_ENCODER_PATH = os.path.join('Text-to-Emotion', 'Model', 'label_encoder.pkl')

# --- Load Models ---
print("Loading models...")
try:
    video_model = load_model(VIDEO_MODEL_PATH)
    print("Video model loaded.")
except Exception as e:
    print(f"Error loading video model: {e}")
    video_model = None

try:
    voice_model = tf.keras.models.load_model(VOICE_MODEL_PATH)
    print("Voice model loaded.")
except Exception as e:
    print(f"Error loading voice model: {e}")
    voice_model = None

try:
    text_model = load_model(TEXT_MODEL_PATH)
    vectorizer = joblib.load(VECTORIZER_PATH)
    label_encoder = joblib.load(LABEL_ENCODER_PATH)
    print("Text model and artifacts loaded.")
except Exception as e:
    print(f"Error loading text model: {e}")
    text_model = None
    vectorizer = None
    label_encoder = None

# --- Constants ---
VIDEO_EMOTION_LABELS = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
VOICE_EMOTION_LABELS = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']

# --- Global State for Video ---
video_emotion_scores = np.zeros(len(VIDEO_EMOTION_LABELS))
video_start_time = None
video_processing = False
cap = None

# --- Helper Functions ---

def get_video_capture():
    global cap
    if cap is None or not cap.isOpened():
        cap = cv2.VideoCapture(0)
    return cap

def generate_frames():
    global video_start_time, video_processing, video_emotion_scores, cap
    
    cap = get_video_capture()
    
    # Reset scores and timer
    video_emotion_scores = np.zeros(len(VIDEO_EMOTION_LABELS))
    video_start_time = time.time()
    video_processing = True
    
    face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

    while video_processing:
        # Stop after 10 seconds
        if time.time() - video_start_time >= 10:
            video_processing = False
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

            if video_model:
                prediction = video_model.predict(roi_gray, verbose=0)
                video_emotion_scores += prediction[0]

        # Encode frame for streaming
        ret, buffer = cv2.imencode('.jpg', frame)
        frame = buffer.tobytes()

        yield (b'--frame\r\n'
               b'Content-Type: image/jpeg\r\n\r\n' + frame + b'\r\n')
    
    # Release camera when done (optional, but good practice if we want to allow other apps to use it)
    # cap.release() 

def record_audio(duration=5, fs=16000, filename="temp_audio.wav"):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording finished.")
    wav.write(filename, fs, recording)
    return filename

def extract_voice_features(audio_path):
    try:
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return mfccs.mean(axis=1)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def get_color_for_emotion(emotion):
    """
    Maps an emotion string to a hex color code.
    Handles various capitalizations and synonyms.
    """
    emotion = emotion.lower().strip()
    
    color_map = {
        # Anger
        'anger': '#FF4500',     # OrangeRed
        'angry': '#FF4500',
        'disgust': '#556B2F',   # DarkOliveGreen
        
        # Fear
        'fear': '#800080',      # Purple
        'fearful': '#800080',
        
        # Happy/Joy
        'happy': '#FFD700',     # Gold
        'joy': '#FFD700',
        
        # Sadness
        'sadness': '#1E90FF',   # DodgerBlue
        'sad': '#1E90FF',
        
        # Surprise
        'surprise': '#00CED1',  # DarkTurquoise
        'surprised': '#00CED1',
        
        # Love
        'love': '#FF69B4',      # HotPink
        
        # Neutral/Calm
        'neutral': '#808080',   # Grey
        'calm': '#ADD8E6'       # LightBlue
    }
    
    return color_map.get(emotion, '#808080') # Default to Grey if unknown

# --- Routes ---

@app.route('/', methods=['GET', 'POST'])
def index():
    """
    Main page. Handles text input (existing) and links to other modules.
    """
    user_text = ""
    emotion = None
    color = None

    if request.method == 'POST':
        user_text = request.form.get('user_text', '')
        # Placeholder for text emotion detection
        if user_text and text_model:
            try:
                # Preprocess text
                # Note: The training script used TfidfVectorizer which returns a sparse matrix or array.
                # The model expects input shape (None, 10000).
                text_vectorized = vectorizer.transform([user_text]).toarray()
                
                # Predict
                prediction = text_model.predict(text_vectorized, verbose=0)
                predicted_index = np.argmax(prediction)
                emotion = label_encoder.inverse_transform([predicted_index])[0]
                
                # Map emotion to color
                color = get_color_for_emotion(emotion)
                
            except Exception as e:
                print(f"Error in text prediction: {e}")
                emotion = "Error"
                color = "#000000"
        elif user_text:
             # Fallback if model not loaded
            emotion = "Model not loaded"
            color = "#808080"

    return render_template('index.html', user_text=user_text, emotion=emotion, color=color)

@app.route('/video_feed')
def video_feed():
    return Response(generate_frames(), mimetype='multipart/x-mixed-replace; boundary=frame')

@app.route('/video_result')
def video_result():
    if np.sum(video_emotion_scores) == 0:
        return jsonify({"error": "No faces detected or processing not started."})

    final_index = np.argmax(video_emotion_scores)
    dominant_emotion = VIDEO_EMOTION_LABELS[final_index]

    # Adjustments from original code
    if dominant_emotion == 'Surprise':
        dominant_emotion = 'Happy'
    elif dominant_emotion == 'Disgust':
        dominant_emotion = 'Angry'

    return jsonify({
        "dominant_emotion": dominant_emotion,
        "score": float(video_emotion_scores[final_index]),
        "color": get_color_for_emotion(dominant_emotion)
    })

@app.route('/voice_predict', methods=['POST'])
def voice_predict():
    if not voice_model:
        return jsonify({"error": "Voice model not loaded."})

    try:
        # Record audio
        filename = record_audio()
        
        # Predict
        feature = extract_voice_features(filename)
        if feature is None:
             return jsonify({"error": "Could not extract features."})
        
        feature = np.expand_dims(feature, axis=0)
        feature = np.expand_dims(feature, axis=-1)
        
        prediction = voice_model.predict(feature, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_emotion = VOICE_EMOTION_LABELS[predicted_index]
        confidence = float(prediction[0][predicted_index])
        
        # Cleanup
        if os.path.exists(filename):
            os.remove(filename)
            
        return jsonify({
            "emotion": predicted_emotion,
            "confidence": confidence,
            "color": get_color_for_emotion(predicted_emotion)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    app.run(debug=True)
