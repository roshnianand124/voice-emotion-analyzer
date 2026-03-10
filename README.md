# VocalSense: AI Voice Emotion Analyzer

## Project Overview

VocalSense is a speech emotion recognition system that detects human emotions from voice recordings using machine learning and audio signal processing. The system analyzes speech signals, extracts audio features, and predicts the emotional state expressed in the voice.

This project demonstrates the use of machine learning, signal processing, and interactive web applications for emotion detection.

---

## Features

* Detects emotions from voice recordings
* Extracts audio features such as MFCC, chroma, and spectral contrast
* Uses a trained machine learning model for classification
* Interactive web interface for uploading audio files
* Displays audio waveform visualization

---

## Technologies Used

* Python
* Librosa
* NumPy
* Scikit-learn
* Streamlit
* Matplotlib

---

## Dataset

This project uses the **RAVDESS (Ryerson Audio-Visual Database of Emotional Speech and Song)** dataset.

The dataset contains speech recordings of actors expressing different emotions such as:

* Neutral
* Calm
* Happy
* Sad
* Angry
* Fearful
* Disgust
* Surprised

---

## Project Structure

voice_emotion_analyzer
│
├── dataset
│   └── ravdess
│       ├── Actor_01
│       ├── Actor_02
│       └── ...
│
├── train_model.py
├── app.py
├── emotion_model.pkl
└── README.md

---

## Installation

Install required libraries using:

pip install librosa numpy pandas scikit-learn matplotlib streamlit soundfile

---

## Training the Model

Run the training script:

python train_model.py

This will train the model and generate the file:

emotion_model.pkl

---

## Running the Web Application

Start the Streamlit application:

python -m streamlit run app.py

Then open the local URL shown in the terminal (usually http://localhost:8501).

---

## How It Works

1. Upload a `.wav` audio file.
2. The system extracts audio features from the speech signal.
3. The trained machine learning model predicts the emotion.
4. The predicted emotion and waveform visualization are displayed.

---

## Applications

* Human-computer interaction
* Emotion-aware AI assistants
* Mental health monitoring
* Call center sentiment analysis
* Speech analytics

---

## Author

Roshni Anand
B.Tech Student – Vellore Institute of Technology
