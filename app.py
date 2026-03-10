import streamlit as st
import librosa
import numpy as np
import pickle
import tempfile
import matplotlib.pyplot as plt

# Load trained model
model = pickle.load(open("emotion_model.pkl", "rb"))

# Feature extraction
def extract_features(file_path):

    audio, sr = librosa.load(file_path, sr=None)

    mfcc = np.mean(librosa.feature.mfcc(y=audio, sr=sr, n_mfcc=40).T, axis=0)
    chroma = np.mean(librosa.feature.chroma_stft(y=audio, sr=sr).T, axis=0)
    contrast = np.mean(librosa.feature.spectral_contrast(y=audio, sr=sr).T, axis=0)

    features = np.hstack([mfcc, chroma, contrast])

    return features, audio


st.title("VocalSense: AI Voice Emotion Analyzer")

st.write("Upload a WAV file to detect emotion from voice.")

uploaded_file = st.file_uploader("Upload WAV file", type=["wav"])

if uploaded_file is not None:

    st.audio(uploaded_file)

    # Save uploaded file temporarily
    with tempfile.NamedTemporaryFile(delete=False, suffix=".wav") as tmp:
        tmp.write(uploaded_file.read())
        temp_path = tmp.name

    features, audio = extract_features(temp_path)

    # Predict emotion
    prediction = model.predict([features])

    st.success("Detected Emotion: " + prediction[0])

    # Plot waveform
    st.subheader("Audio Waveform")

    fig, ax = plt.subplots()

    ax.plot(audio)

    ax.set_xlabel("Time")

    ax.set_ylabel("Amplitude")

    st.pyplot(fig)
