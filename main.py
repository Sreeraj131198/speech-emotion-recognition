"""
FastAPI Deployment for Speech Emotion Recognition Model
CNN-LSTM Hybrid Model with librosa feature extraction
"""

import os
import io
import pickle
import numpy as np
import librosa
import tensorflow as tf
from fastapi import FastAPI, File, UploadFile, HTTPException
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
from typing import Optional
import uvicorn

# ─────────────────────────────────────────────
# App setup
# ─────────────────────────────────────────────
app = FastAPI(
    title="Speech Emotion Recognition API",
    description="Upload a WAV audio file and get the predicted emotion using a CNN-LSTM deep learning model.",
    version="1.0.0",
)

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ─────────────────────────────────────────────
# Load model, scaler, label encoder at startup
# ─────────────────────────────────────────────
MODEL_PATH        = os.getenv("MODEL_PATH",         "emotion_recognition_model.h5")
SCALER_PATH       = os.getenv("SCALER_PATH",        "scaler.pkl")
LABEL_ENCODER_PATH = os.getenv("LABEL_ENCODER_PATH","label_encoder.pkl")

model         = None
scaler        = None
label_encoder = None


@app.on_event("startup")
def load_artifacts():
    global model, scaler, label_encoder

    if not os.path.exists(MODEL_PATH):
        raise RuntimeError(f"Model file not found: {MODEL_PATH}")
    if not os.path.exists(SCALER_PATH):
        raise RuntimeError(f"Scaler file not found: {SCALER_PATH}")
    if not os.path.exists(LABEL_ENCODER_PATH):
        raise RuntimeError(f"Label encoder file not found: {LABEL_ENCODER_PATH}")

    model = tf.keras.models.load_model(MODEL_PATH)
    with open(SCALER_PATH, "rb") as f:
        scaler = pickle.load(f)
    with open(LABEL_ENCODER_PATH, "rb") as f:
        label_encoder = pickle.load(f)

    print("✅ Model, scaler, and label encoder loaded successfully.")
    print(f"   Emotions: {list(label_encoder.classes_)}")


# ─────────────────────────────────────────────
# Feature extraction (mirrors notebook exactly)
# ─────────────────────────────────────────────
def extract_features(audio_bytes: bytes, duration: int = 3, sr: int = 22050) -> Optional[np.ndarray]:
    """
    Extract the same features used during training:
    MFCCs (40) | Chroma | Mel Spectrogram | Spectral Contrast | Tonnetz
    Returns a 1-D numpy array or None on error.
    """
    try:
        audio_io = io.BytesIO(audio_bytes)
        audio, sample_rate = librosa.load(audio_io, duration=duration, sr=sr)

        # MFCCs
        mfccs        = librosa.feature.mfcc(y=audio, sr=sample_rate, n_mfcc=40)
        mfccs_mean   = np.mean(mfccs.T, axis=0)
        mfccs_std    = np.std(mfccs.T, axis=0)

        # Chroma
        chroma       = librosa.feature.chroma_stft(y=audio, sr=sample_rate)
        chroma_mean  = np.mean(chroma.T, axis=0)
        chroma_std   = np.std(chroma.T, axis=0)

        # Mel Spectrogram
        mel          = librosa.feature.melspectrogram(y=audio, sr=sample_rate)
        mel_mean     = np.mean(mel.T, axis=0)
        mel_std      = np.std(mel.T, axis=0)

        # Spectral Contrast
        contrast      = librosa.feature.spectral_contrast(y=audio, sr=sample_rate)
        contrast_mean = np.mean(contrast.T, axis=0)
        contrast_std  = np.std(contrast.T, axis=0)

        # Tonnetz
        tonnetz       = librosa.feature.tonnetz(y=audio, sr=sample_rate)
        tonnetz_mean  = np.mean(tonnetz.T, axis=0)
        tonnetz_std   = np.std(tonnetz.T, axis=0)

        features = np.concatenate([
            mfccs_mean,   mfccs_std,
            chroma_mean,  chroma_std,
            mel_mean,     mel_std,
            contrast_mean, contrast_std,
            tonnetz_mean, tonnetz_std,
        ])
        return features

    except Exception as e:
        print(f"Feature extraction error: {e}")
        return None


# ─────────────────────────────────────────────
# Response schemas
# ─────────────────────────────────────────────
class EmotionResponse(BaseModel):
    predicted_emotion: str
    confidence: float
    confidence_pct: str
    all_probabilities: dict[str, float]


class HealthResponse(BaseModel):
    status: str
    model_loaded: bool
    supported_emotions: list[str]


# ─────────────────────────────────────────────
# Endpoints
# ─────────────────────────────────────────────
@app.get("/", tags=["Root"])
def root():
    return {"message": "Speech Emotion Recognition API is running. POST an audio file to /predict"}


@app.get("/health", response_model=HealthResponse, tags=["Health"])
def health():
    return HealthResponse(
        status="ok" if model is not None else "model not loaded",
        model_loaded=model is not None,
        supported_emotions=list(label_encoder.classes_) if label_encoder else [],
    )


@app.post("/predict", response_model=EmotionResponse, tags=["Prediction"])
async def predict(file: UploadFile = File(..., description="WAV audio file (mono or stereo)")):
    # Validate file type
    if not file.filename.lower().endswith((".wav", ".mp3", ".flac", ".ogg")):
        raise HTTPException(
            status_code=400,
            detail="Unsupported file type. Please upload a .wav, .mp3, .flac, or .ogg file.",
        )

    # Read file bytes
    audio_bytes = await file.read()
    if len(audio_bytes) == 0:
        raise HTTPException(status_code=400, detail="Uploaded file is empty.")

    # Extract features
    features = extract_features(audio_bytes)
    if features is None:
        raise HTTPException(
            status_code=422,
            detail="Could not extract features from the audio. Ensure it is a valid audio file.",
        )

    # Preprocess: scale → reshape for Conv1D (1, features, 1)
    features = features.reshape(1, -1)
    features = scaler.transform(features)
    features = np.expand_dims(features, axis=2)

    # Predict
    probabilities  = model.predict(features, verbose=0)[0]
    predicted_idx  = int(np.argmax(probabilities))
    predicted_label = label_encoder.inverse_transform([predicted_idx])[0]
    confidence      = float(probabilities[predicted_idx])

    all_probs = {
        label_encoder.classes_[i]: round(float(probabilities[i]), 4)
        for i in range(len(label_encoder.classes_))
    }

    return EmotionResponse(
        predicted_emotion=predicted_label,
        confidence=round(confidence, 4),
        confidence_pct=f"{confidence * 100:.2f}%",
        all_probabilities=all_probs,
    )


# ─────────────────────────────────────────────
# Run directly
# ─────────────────────────────────────────────
if __name__ == "__main__":
    uvicorn.run("main:app", host="0.0.0.0", port=8000, reload=False)
