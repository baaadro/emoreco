#this code is for untrained model, keep in mind it trains the model as a first step so it will take time


import numpy as np
import pandas as pd
import cv2
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Conv2D, MaxPooling2D, Flatten, Dense, Dropout, BatchNormalization
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical
from sklearn.model_selection import train_test_split
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# data preprocessing using (FER-2013) lien: https://www.kaggle.com/datasets/nicolejyt/facialexpressionrecognition?resource=download

def load_fer2013():
    data = pd.read_csv("fer2013.csv")  
    pixels = data['pixels'].apply(lambda x: np.array(x.split(' '), dtype='float32'))
    X = np.vstack(pixels.values).reshape(-1, 48, 48, 1)  
    y = to_categorical(data['emotion'])  
    return train_test_split(X, y, test_size=0.2, random_state=42)

X_train, X_test, y_train, y_test = load_fer2013()
X_train, X_val, y_train, y_val = train_test_split(X_train, y_train, test_size=0.1, random_state=42)

# normalization
X_train = X_train / 255.0
X_val = X_val / 255.0
X_test = X_test / 255.0

datagen = ImageDataGenerator(rotation_range=15, zoom_range=0.2, horizontal_flip=True)
datagen.fit(X_train)

# building the main cnn model 
model = Sequential([
    Conv2D(64, (3,3), activation='relu', input_shape=(48, 48, 1)),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(128, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Conv2D(256, (3,3), activation='relu'),
    BatchNormalization(),
    MaxPooling2D((2,2)),
    Flatten(),
    Dense(512, activation='relu'),
    Dropout(0.5),
    Dense(7, activation='softmax')  # 7 emotions
])

model.compile(optimizer=Adam(learning_rate=0.001), 
              loss='categorical_crossentropy', 
              metrics=['accuracy'])

# model training
history = model.fit(datagen.flow(X_train, y_train, batch_size=64),
                    validation_data=(X_val, y_val),
                    epochs=20,
                    verbose=1)


# testing
test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
print(f"Test Accuracy: {test_acc:.4f}")

# training history (optional)
plt.plot(history.history['accuracy'], label='Training Accuracy')
plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
plt.xlabel('Epoch')
plt.ylabel('Accuracy')
plt.legend()
plt.show()

# model save (optional)
model.save("emotion_model.h5")  # Saves the trained model
print("Model saved as 'emotion_model.h5'")

# face and emotion detection using real time webcam (opencv)
face_cascade = cv2.CascadeClassifier(cv2.data.haarcascades + 'haarcascade_frontalface_default.xml')

def detect_emotion(frame):
    gray = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(gray, 1.1, 5)
    
    for (x, y, w, h) in faces:
        face_roi = gray[y:y+h, x:x+w]
        face_roi = cv2.resize(face_roi, (48, 48))
        face_roi = face_roi.reshape(1, 48, 48, 1) / 255.0
        
        # emotion prediction
        emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
        pred = model.predict(face_roi)
        emotion = emotion_labels[np.argmax(pred)]
        
        cv2.rectangle(frame, (x, y), (x+w, y+h), (255, 0, 0), 2)
        cv2.putText(frame, emotion, (x, y-10), cv2.FONT_HERSHEY_SIMPLEX, 0.9, (0, 255, 0), 2)
    
    return frame

# webcam
cap = cv2.VideoCapture(0)
while True:
    ret, frame = cap.read()
    if not ret:
        break
    
    frame = detect_emotion(frame)
    cv2.imshow('Emotion Detection', frame)
    # quit using q
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()


#Image statique
#ajouter la photo au directoire, nommer photo1.jpg
emotion_labels = ['Angry', 'Disgust', 'Fear', 'Happy', 'Sad', 'Surprise', 'Neutral']
img = cv2.imread("photo1.png")
emotion = detect_emotion(img)
print(f"Émotion prédite : {emotion_labels[emotion]}")
