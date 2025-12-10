import os
import time
import cv2
import numpy as np
import tensorflow as tf
from flask import Flask, render_template, request, Response, jsonify
from flask_cors import CORS
from tensorflow.keras.models import load_model
# import sounddevice as sd # Removed for client-side recording
import scipy.io.wavfile as wav
import librosa
import joblib
from tensorflow.keras.preprocessing.sequence import pad_sequences
from pydub import AudioSegment
import imageio_ffmpeg

import logging

# Configure logging
logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')
logger = logging.getLogger(__name__)

# Configure pydub to use imageio-ffmpeg's binary
AudioSegment.converter = imageio_ffmpeg.get_ffmpeg_exe()

app = Flask(__name__)
CORS(app) # Enable CORS for all routes

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

# Server-side recording functions removed
# def record_audio...
# def get_video_capture...
# def generate_frames... 

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
    logger.info(f"Connection received from {request.remote_addr} to /")
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

@app.route('/text_predict', methods=['POST'])
def text_predict():
    logger.info(f"Connection received from {request.remote_addr} to /text_predict")
    try:
        data = request.get_json()
        user_text = data.get('text', '')
        
        if not user_text:
             return jsonify({"error": "No text provided"}), 400

        if not text_model:
             return jsonify({"error": "Text model not loaded"}), 503

        # Preprocess text
        text_vectorized = vectorizer.transform([user_text]).toarray()
        
        # Predict
        prediction = text_model.predict(text_vectorized, verbose=0)
        predicted_index = np.argmax(prediction)
        emotion = label_encoder.inverse_transform([predicted_index])[0]
        
        # Map emotion to color
        color = get_color_for_emotion(emotion)
        
        logger.info(f"Text prediction: {emotion} (Color: {color})")
        return jsonify({
            "emotion": emotion,
            "color": color
        })
                
    except Exception as e:
        print(f"Error in text prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/video_frame_predict', methods=['POST'])
def video_frame_predict():
    logger.info(f"Connection received from {request.remote_addr} to /video_frame_predict")
    if not video_model:
        return jsonify({"error": "Video model not loaded"}), 503

    try:
        file = request.files.get('frame')
        if not file:
            return jsonify({"error": "No frame provided"}), 400

        # Decode image
        npimg = np.frombuffer(file.read(), np.uint8)
        frame = cv2.imdecode(npimg, cv2.IMREAD_COLOR)
        
        if frame is None:
             return jsonify({"error": "Could not decode image"}), 400

        gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')
        faces = face_cascade.detectMultiScale(gray, 1.3, 5)

        if len(faces) == 0:
            return jsonify({"status": "no_face"})

        # Process the first face found
        (x, y, w, h) = faces[0]
        roi_gray = gray[y:y + h, x:x + w]
        roi_gray = cv2.resize(roi_gray, (48, 48))
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)

        prediction = video_model.predict(roi_gray, verbose=0)
        
        # Return raw scores for client-side aggregation
        logger.info(f"Video frame processed. Scores: {prediction[0].tolist()}")
        return jsonify({
            "status": "success",
            "scores": prediction[0].tolist()
        })

    except Exception as e:
        print(f"Error in video frame prediction: {e}")
        return jsonify({"error": str(e)}), 500

@app.route('/voice_predict', methods=['POST'])
def voice_predict():
    logger.info(f"Connection received from {request.remote_addr} to /voice_predict")
    if not voice_model:
        return jsonify({"error": "Voice model not loaded."})

    try:
        if 'file' not in request.files:
            return jsonify({"error": "No file part"}), 400
        
        file = request.files['file']
        if file.filename == '':
            return jsonify({"error": "No selected file"}), 400

        # Save temporarily (original format)
        original_filename = "temp_upload" + os.path.splitext(file.filename)[1]
        wav_filename = "temp_upload.wav"
        file.save(original_filename)
        
        try:
            # Convert to WAV using ffmpeg directly via subprocess
            import subprocess
            ffmpeg_exe = imageio_ffmpeg.get_ffmpeg_exe()
            
            # -y: overwrite output files
            # -i: input file
            # output file is the last argument
            command = [ffmpeg_exe, '-y', '-i', original_filename, wav_filename]
            
            # Run conversion, suppress output
            subprocess.run(command, check=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
            
            # Predict using the converted WAV file
            feature = extract_voice_features(wav_filename)
            
        except Exception as conversion_error:
            print(f"Conversion error: {conversion_error}")
            # Fallback: try using the original file if conversion fails
            feature = extract_voice_features(original_filename)

        # Cleanup
        if os.path.exists(original_filename):
            os.remove(original_filename)
        if os.path.exists(wav_filename):
            os.remove(wav_filename)

        if feature is None:
             return jsonify({"error": "Could not extract features. Ensure file is a valid audio file."})
        
        feature = np.expand_dims(feature, axis=0)
        feature = np.expand_dims(feature, axis=-1)
        
        prediction = voice_model.predict(feature, verbose=0)
        predicted_index = np.argmax(prediction)
        predicted_emotion = VOICE_EMOTION_LABELS[predicted_index]
        confidence = float(prediction[0][predicted_index])
            
        logger.info(f"Voice prediction: {predicted_emotion} (Confidence: {confidence:.2f})")
        return jsonify({
            "emotion": predicted_emotion,
            "confidence": confidence,
            "color": get_color_for_emotion(predicted_emotion)
        })

    except Exception as e:
        return jsonify({"error": str(e)})

if __name__ == '__main__':
    # Run on 0.0.0.0 to allow external access (e.g., from EC2 public IP)
    app.run(host='0.0.0.0', port=5000, debug=True)
