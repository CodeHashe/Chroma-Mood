import cv2
import numpy as np
from tensorflow.keras.models import load_model
from collections import Counter
import time

start_time = time.time() 


model = load_model('emotionsvideomodel.h5')

emotion_labels = ['Anger', 'Disgust', 'Fear', 'Happy', 'Neutral', 'Sadness', "Surprise"]
emotion_scores = np.zeros(len(emotion_labels))

face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')


cap = cv2.VideoCapture(0)  # 0 = default webcam


while True:
    if time.time() - start_time >= 10:
        break

    ret, frame = cap.read()
    if not ret:
        break

    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, scaleFactor=1.3, minNeighbors=5)

    for (x, y, w, h) in faces:
         # Draw rectangle around the face
        cv2.rectangle(frame, (x, y), (x + w, y + h), (0, 255, 0), 2)
        roi_gray = gray[y:y+h, x:x+w]             # Crop the face
        roi_gray = cv2.resize(roi_gray, (48, 48)) # FER2013 images are 48x48
        roi_gray = roi_gray.astype('float') / 255.0
        roi_gray = np.expand_dims(roi_gray, axis=0)
        roi_gray = np.expand_dims(roi_gray, axis=-1)  # Add channel dimension

        prediction = model.predict(roi_gray)
        emotion_scores += prediction[0]
        max_index = int(np.argmax(prediction))
        emotion = emotion_labels[max_index]
        print(emotion)
    cv2.imshow('Emotion Detector', frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):  # Press 'q' to quit
        break

cap.release()
cv2.destroyAllWindows()

if np.sum(emotion_scores) > 0:   # Means at least one prediction happened
    final_index = np.argmax(emotion_scores)
    dominant_emotion = emotion_labels[final_index]

    # Fix common FER typo or confusion
    if dominant_emotion == 'Surprise':   # note correct spelling
        dominant_emotion = 'Happy'
    elif dominant_emotion =='Disgust':
        dominant_emotion='Angry'
    print("Emotion Detected:", dominant_emotion,emotion_scores[final_index])
else:
    print("No faces detected.")
