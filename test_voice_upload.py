import requests
import numpy as np
import scipy.io.wavfile as wav
import os

# Create a dummy WAV file
sample_rate = 16000
duration = 1 # second
t = np.linspace(0, duration, int(sample_rate * duration), endpoint=False)
audio = 0.5 * np.sin(2 * np.pi * 440 * t) # 440 Hz sine wave
filename = "test_audio.wav"
wav.write(filename, sample_rate, (audio * 32767).astype(np.int16))

url = "http://127.0.0.1:5000/voice_predict"
files = {'file': open(filename, 'rb')}

try:
    print(f"Sending {filename} to {url}...")
    response = requests.post(url, files=files)
    print(f"Status Code: {response.status_code}")
    print(f"Response: {response.text}")
except Exception as e:
    print(f"Error: {e}")
finally:
    files['file'].close()
    if os.path.exists(filename):
        os.remove(filename)
