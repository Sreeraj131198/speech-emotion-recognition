# 🎙️ Speech Emotion Recognition (SER)

## Overview

This project focuses on building a deep learning model capable of identifying the **emotion conveyed in a voice recording**. The model classifies audio clips into distinct emotional categories using acoustic features extracted from speech signals.

---

## 📂 Datasets Used

The project combines four widely-used, publicly available speech emotion datasets into a single unified dataframe:

| Dataset | Description |
|--------|-------------|
| **CREMA-D** | Crowd-sourced Emotional Multimodal Actors Dataset |
| **SAVEE** | Surrey Audio-Visual Expressed Emotion |
| **TESS** | Toronto Emotional Speech Set |
| **RAVDESS** | Ryerson Audio-Visual Database of Emotional Speech and Song |

> Each dataset uses a different file naming convention, which was handled during preprocessing to unify them into a single dataframe.

---

## 🔁 Project Workflow

### 1. Data Loading
- Downloaded and loaded all four datasets.
- Carefully parsed each dataset's unique naming convention to extract emotion labels.

### 2. Preprocessing
- Merged all datasets into a **single unified DataFrame** with consistent label encoding across datasets.

### 3. Exploratory Data Analysis (EDA)
- **Spectrograms**: Visual representation of the frequency spectrum of audio signals over time.
- **Waveplots**: Amplitude vs. time plots showing the waveform of audio signals.
- **Count Plot**: Distribution of emotion classes across the combined dataset to identify class imbalance.

### 4. Data Augmentation
To improve model generalization, the following augmentation techniques were applied to the audio data:

| Technique | Description |
|-----------|-------------|
| **Stretching** | Time-stretches the audio signal without altering pitch, making the model robust to speaking rate variations. |
| **Noise Injection** | Adds random Gaussian noise to the audio signal, simulating real-world recording conditions. |
| **Pitch Shifting** | Shifts the pitch of the audio up or down while maintaining the original duration, helping the model generalize across different voices. |

### 5. Feature Extraction
Audio features were extracted using the **[Librosa](https://librosa.org/)** library.

#### Initial Features (Baseline Model)
| Feature | Description |
|---------|-------------|
| **MFCC** (Mel-Frequency Cepstral Coefficients) | Captures the short-term power spectrum of sound; one of the most widely used features in speech processing. Represents how the human ear perceives frequencies on a non-linear mel scale. |
| **RMSE** (Root Mean Square Energy) | Measures the loudness/energy of the audio signal over time. Useful for detecting silent vs. active speech regions. |

#### Improved Features (Enhanced Model)
| Feature | Description |
|---------|-------------|
| **MFCC** | Same as above |
| **Chroma** | Represents the 12 different pitch classes (C, C#, D, ..., B). Captures harmonic and melodic characteristics of audio, useful for tonal emotion detection. |
| **Tonnetz** | Tonal centroid features that capture harmonic relationships in the audio. Useful for capturing the tonal structure associated with different emotions. |
| **Spectral Contrast** | Measures the difference in amplitude between peaks and valleys in a sound spectrum. Helps distinguish between harmonically rich (e.g., happy) and flat (e.g., sad/neutral) sounds. |

### 6. Model Training

#### Baseline Model
- Architecture: **1D Convolutional Neural Network (Conv1D)**
- Features: MFCC + RMSE only
- Result: ❌ **~18% accuracy** — poor performance due to limited features and shallow architecture.

#### Improved Model
- Architecture: **Conv1D + Bidirectional LSTM**
- Features: MFCC + Chroma + Tonnetz + Spectral Contrast
- Result: ✅ **~67% accuracy** — significant improvement due to richer features and the LSTM's ability to capture temporal patterns in speech.

> **Why Bidirectional LSTM?** Speech is inherently sequential. A Bidirectional LSTM processes the audio sequence in both forward and backward directions, allowing it to capture context from both past and future time steps — critical for understanding the emotional arc of a spoken sentence.

---

## 🛠️ Libraries & Tools

- **[Librosa](https://librosa.org/)** — Audio feature extraction
- **NumPy / Pandas** — Data manipulation
- **Matplotlib / Seaborn** — Data visualization
- **TensorFlow / Keras** — Model building and training
- **Scikit-learn** — Data splitting and evaluation metrics

---

## 📈 Results Summary

| Model | Features | Architecture | Accuracy |
|-------|----------|--------------|----------|
| Baseline | MFCC, RMSE | Conv1D | ~18% |
| Improved | MFCC, Chroma, Tonnetz, Spectral Contrast | Conv1D + Bidirectional LSTM | ~67% |

---

## 🚀 Future Improvements

- Experiment with deeper architectures (e.g., Transformer-based models)
- Add more augmentation techniques (e.g., speed perturbation, room impulse response)
- Use attention mechanisms to focus on the most emotionally expressive parts of speech
- Increase dataset size or use transfer learning with pre-trained audio models (e.g., wav2vec 2.0)

---

## 📌 Key Takeaways

- Feature richness has a massive impact on model performance — going from 2 to 4 feature types boosted accuracy from 18% to 67%.
- Bidirectional LSTMs are well-suited for sequential audio data, capturing temporal dependencies that CNNs alone cannot.
- Data augmentation helps mitigate overfitting and improves robustness across different speakers and recording conditions.# speech-emotion-recognition
