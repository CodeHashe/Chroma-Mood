# Speech Emotion Recognition (SER) Implementation Plan

This guide outlines the steps to build a Speech Emotion Recognition model using the RAVDESS dataset found in `Datasets/Audio_Speech_Actors_01-24`.

## Step 1: Environment Setup
You will need the following Python libraries:
- **librosa**: For audio processing and feature extraction.
- **soundfile**: For reading/writing audio files.
- **numpy**: For numerical operations.
- **pandas**: For data handling.
- **scikit-learn**: For data splitting and label encoding.
- **tensorflow** (or keras): For building and training the neural network.

**Installation:**
```bash
pip install librosa soundfile numpy pandas scikit-learn tensorflow
```

## Step 2: Data Loading & Label Extraction
The RAVDESS dataset files follow a specific naming convention: `03-01-01-01-01-01-01.wav`.
The **3rd identifier** represents the emotion:
- 01 = neutral
- 02 = calm
- 03 = happy
- 04 = sad
- 05 = angry
- 06 = fearful
- 07 = disgust
- 08 = surprised

**Action:**
1. Traverse the `Datasets/Audio_Speech_Actors_01-24` directory.
2. For each `.wav` file, extract the emotion code from the filename.
3. Store the file path and the corresponding emotion label in a list or DataFrame.

## Step 3: Feature Extraction (MFCCs)
We need to convert raw audio into numerical features. Mel-frequency cepstral coefficients (MFCCs) are commonly used for speech.

**Action:**
1. Use `librosa.load(file_path)` to load the audio.
2. Use `librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)` to extract features.
3. Compute the **mean** of the MFCCs across the time axis to get a single vector per audio file (e.g., shape `(40,)`).

## Step 4: Data Preparation
1. **X (Features):** Convert your list of MFCC vectors into a numpy array.
2. **y (Labels):** Convert emotion labels into a format suitable for training (e.g., One-Hot Encoding using `pd.get_dummies` or `to_categorical`).
3. **Split:** Use `train_test_split` from sklearn to create training and testing sets.
4. **Reshape:** If using a CNN, reshape X to have a 3rd dimension: `(num_samples, 40, 1)`.

## Step 5: Model Building
Build a Neural Network (CNN or LSTM is recommended).

**Example CNN Architecture:**
- **Conv1D Layer**: 64 filters, kernel size 5, activation 'relu'.
- **MaxPooling1D**: Pool size 2.
- **Dropout**: 0.2 (to prevent overfitting).
- **Conv1D Layer**: 128 filters, kernel size 5, activation 'relu'.
- **MaxPooling1D**: Pool size 2.
- **Dropout**: 0.2.
- **Flatten Layer**.
- **Dense Layer**: 64 units, activation 'relu'.
- **Output Layer**: Units = number of emotions (8), activation 'softmax'.

## Step 6: Training
1. **Compile:** Use `loss='categorical_crossentropy'`, `optimizer='adam'`, `metrics=['accuracy']`.
2. **Fit:** Train the model on the training data (e.g., 50 epochs).
3. **Save:** Save the trained model using `model.save('emotion_model.h5')`.

## Step 7: Real-time Prediction
To detect emotion in a user's voice:
1. Record or load a new audio clip.
2. Preprocess it exactly like the training data (load -> extract MFCCs -> mean -> reshape).
3. Use `model.predict()` to get probabilities.
4. Map the highest probability index back to the emotion label.
