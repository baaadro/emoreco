
#this code is only usable if you have the trained model



import cv2
import numpy as np
from tensorflow.keras.models import load_model

# Load pre-trained model and face detector
model = load_model("emotion_model.h5")  # Replace with your path
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

# Emotion labels
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']

def preprocess_face(face_roi):
    """Resize and normalize face ROI for the model"""
    face_roi = cv2.resize(face_roi, (48, 48))
    face_roi = cv2.cvtColor(face_roi, cv2.COLOR_BGR2GRAY)
    face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
    return face_roi

# Real-time detection
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        face_roi = frame[y:y+h, x:x+w]
        processed_face = preprocess_face(face_roi)
        pred = model.predict(processed_face)
        emotion = emotion_labels[np.argmax(pred)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    cv2.imshow('Emotion Detection', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
