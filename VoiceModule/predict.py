import sounddevice as sd
import scipy.io.wavfile as wav
import librosa
import numpy as np
import tensorflow as tf
import os

def record_audio(duration=5, fs=16000, filename="output.wav"):
    print("Recording...")
    recording = sd.rec(int(duration * fs), samplerate=fs, channels=1)
    sd.wait()
    print("Recording finished.")
    wav.write(filename, fs, recording)
    return filename

def extract_features(audio_path):
    try:
        audio, sample_rate = librosa.load(audio_path, sr=16000)
        mfccs = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        return mfccs.mean(axis=1)
    except Exception as e:
        print(f"Error processing audio: {e}")
        return None

def predict_emotion(model_path, audio_path):
    model = tf.keras.models.load_model(model_path)
    
    feature = extract_features(audio_path)
    if feature is None:
        return
    
    feature = np.expand_dims(feature, axis=0)
    feature = np.expand_dims(feature, axis=-1)
    
    prediction = model.predict(feature)
    predicted_index = np.argmax(prediction)
    
    emotions = ['angry', 'calm', 'disgust', 'fearful', 'happy', 'neutral', 'sad', 'surprised']
    
    predicted_emotion = emotions[predicted_index]
    confidence = prediction[0][predicted_index]
    
    print(f"\nPredicted Emotion: {predicted_emotion.upper()}")
    print(f"Confidence: {confidence:.2f}")
    
    print("\nProbabilities:")
    for i, emotion in enumerate(emotions):
        print(f"{emotion}: {prediction[0][i]:.4f}")

if __name__ == "__main__":
    if not os.path.exists("emotion_model.keras"):
        print("Error: emotion_model.keras not found. Please train the model first.")
    else:
        filename = record_audio()
        predict_emotion("emotion_model.keras", filename)
        
        if os.path.exists(filename):
            os.remove(filename)
