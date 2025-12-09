import os
import librosa
import numpy as np
import tensorflow as tf
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from sklearn.metrics import classification_report
import mlflow
import mlflow.tensorflow

def mapping_audio(path):
    mapping = {}

    emotions = {
        "01": "neutral",
        "02": "calm",
        "03": "happy",
        "04": "sad",
        "05": "angry",
        "06": "fearful",
        "07": "disgust",
        "08": "surprised"
    }

    for root, dirs, files in os.walk(path):
       for filename in files:
         if filename.endswith('.wav'):
             full_path = os.path.join(root, filename)
             mapping[full_path] = emotions[filename.split("-")[2]]
    
    return mapping

def noise(data):
    noise_amp = 0.035*np.random.uniform()*np.amax(data)
    data = data + noise_amp*np.random.normal(size=data.shape[0])
    return data

def stretch(data, rate=0.8):
    return librosa.effects.time_stretch(y=data, rate=rate)

def pitch(data, sampling_rate, pitch_factor=0.7):
    return librosa.effects.pitch_shift(y=data, sr=sampling_rate, n_steps=pitch_factor)

def extract_features(data, sample_rate):
    mfccs = librosa.feature.mfcc(y=data, sr=sample_rate, n_mfcc=40)
    return mfccs.mean(axis=1)

def feature_extraction(mapping):
    features = []
    labels = []

    print("Starting feature extraction with augmentation...")
    
    for audio_path, emotion in mapping.items():
        try:
            audio, sample_rate = librosa.load(audio_path, sr=16000)
            
            features.append(extract_features(audio, sample_rate))
            labels.append(emotion)
            
            noise_audio = noise(audio)
            features.append(extract_features(noise_audio, sample_rate))
            labels.append(emotion)
            
            pitch_audio = pitch(audio, sample_rate)
            features.append(extract_features(pitch_audio, sample_rate))
            labels.append(emotion)
            
            stretch_audio = stretch(audio)
            features.append(extract_features(stretch_audio, sample_rate))
            labels.append(emotion)
            
        except Exception as e:
            print(f"Error processing {audio_path}: {e}")
    
    print(f"Feature extraction complete. Total samples: {len(features)}")
    return features, labels
    
def data_preprocessing(features, labels):
    features = np.array(features)
    features = np.expand_dims(features, axis=-1) 
    
    label_encoder = LabelEncoder()
    labels = label_encoder.fit_transform(labels)
    
    X_train, X_test, y_train, y_test = train_test_split(features, labels, test_size=0.2, random_state=42)
    return X_train, X_test, y_train, y_test, label_encoder


def model_building(input_shape):
    model = tf.keras.models.Sequential([
        tf.keras.layers.Input(shape=input_shape),
        
        tf.keras.layers.Conv1D(128, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv1D(256, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.Conv1D(512, 5, padding='same', activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.MaxPooling1D(2),
        tf.keras.layers.Dropout(0.3),
        
        tf.keras.layers.LSTM(128, return_sequences=True),
        tf.keras.layers.LSTM(64),
        
        tf.keras.layers.Dense(64, activation='relu'),
        tf.keras.layers.BatchNormalization(),
        tf.keras.layers.Dropout(0.4),
        
        tf.keras.layers.Dense(8, activation='softmax')
    ])
    return model

def model_training(model, X_train, y_train, X_test, y_test):
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=0.001),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    
    early_stopping = tf.keras.callbacks.EarlyStopping(
        monitor='val_loss',
        patience=10,
        restore_best_weights=True
    )
    
    reduce_lr = tf.keras.callbacks.ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=5,
        min_lr=0.00001
    )
    
    model.fit(
        X_train, y_train, 
        validation_data=(X_test, y_test), 
        epochs=100, 
        batch_size=32,
        callbacks=[early_stopping, reduce_lr]
    )
    return model

def model_evaluation(model, X_test, y_test, label_encoder):
    print("\nEvaluating model on test set...")
    loss, accuracy = model.evaluate(X_test, y_test, verbose=0)
    print(f"Test Loss: {loss:.4f}")
    print(f"Test Accuracy: {accuracy:.4f}")
    
    y_pred = model.predict(X_test)
    y_pred_classes = np.argmax(y_pred, axis=1)
    
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred_classes, target_names=label_encoder.classes_))

def model_prediction(model, features):
    predictions = model.predict(features)
    return predictions

def main():
    # Calculate absolute path to the dataset
    # This assumes the script is in VoiceModule/ and Datasets/ is in the project root
    base_dir = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
    dataset_path = os.path.join(base_dir, "Datasets", "Audio_Speech_Actors_01-24")
    
    print(f"Dataset path: {dataset_path}")
    
    if not os.path.exists(dataset_path):
        print(f"Error: Dataset not found at {dataset_path}")
        return

    mapping = mapping_audio(dataset_path)
    features, labels = feature_extraction(mapping)
    X_train, X_test, y_train, y_test, label_encoder = data_preprocessing(features, labels)
    model = model_building(X_train.shape[1:])
    # MLflow tracking
    mlflow.set_experiment("Voice_Emotion_Detection")
    mlflow.tensorflow.autolog()

    with mlflow.start_run():
        model = model_training(model, X_train, y_train, X_test, y_test)
        model_evaluation(model, X_test, y_test, label_encoder)
        
        # Log the model artifact
        model.save("emotion_model.keras")
        mlflow.log_artifact("emotion_model.keras")

if __name__ == "__main__":
    main()
