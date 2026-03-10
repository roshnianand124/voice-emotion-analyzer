import os
import numpy as np
import librosa
import pickle
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

print("Program started...")

# Feature extraction
def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)

    return np.hstack([mfcc, chroma, contrast])


dataset_path = "dataset/ravdess"

X = []
y = []

emotion_map = {
    "01": "neutral",
    "02": "calm",
    "03": "happy",
    "04": "sad",
    "05": "angry",
    "06": "fearful",
    "07": "disgust",
    "08": "surprised"
}

print("Reading dataset from:", dataset_path)

file_count = 0

for actor in os.listdir(dataset_path):

    actor_folder = os.path.join(dataset_path, actor)

    if os.path.isdir(actor_folder):

        print("Reading:", actor)

        for file in os.listdir(actor_folder):

            if file.endswith(".wav"):

                file_path = os.path.join(actor_folder, file)

                emotion_code = file.split("-")[2]
                emotion = emotion_map[emotion_code]

                features = extract_features(file_path)

                X.append(features)
                y.append(emotion)

                file_count += 1

print("Total audio files processed:", file_count)

X = np.array(X)
y = np.array(y)

print("Training model...")

X_train, X_test, y_train, y_test = train_test_split(
    X, y, test_size=0.2, random_state=42
)

model = RandomForestClassifier(n_estimators=200)

model.fit(X_train, y_train)

predictions = model.predict(X_test)

accuracy = accuracy_score(y_test, predictions)

print("Model Accuracy:", accuracy)

with open("emotion_model.pkl", "wb") as f:
    pickle.dump(model, f)

print("Model saved as emotion_model.pkl")

