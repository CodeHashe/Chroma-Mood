import tensorflow as tf
import os

model_path = 'emotionvoicemodel.keras'

print(f"Checking if file exists: {os.path.exists(model_path)}")

try:
    model = tf.keras.models.load_model(model_path)
    print("Model loaded successfully!")
except Exception as e:
    print(f"Error loading model: {e}")
