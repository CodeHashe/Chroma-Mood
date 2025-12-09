import tensorflow as tf
import os

model_path_keras = 'emotionvoicemodel.keras'
model_path_h5 = 'emotionvoicemodel.h5'

print(f"Checking .keras: {os.path.exists(model_path_keras)}")
try:
    model = tf.keras.models.load_model(model_path_keras)
    print("Keras model loaded successfully!")
except Exception as e:
    print(f"Error loading .keras model: {e}")

print("-" * 20)

print(f"Checking .h5: {os.path.exists(model_path_h5)}")
try:
    model = tf.keras.models.load_model(model_path_h5)
    print("H5 model loaded successfully!")
except Exception as e:
    print(f"Error loading .h5 model: {e}")
