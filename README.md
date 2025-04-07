# Deep Learning Projects with TensorFlow & Keras

This repository contains four deep learning projects implemented using TensorFlow and Keras. Each project explores a different neural network architecture and application, from image reconstruction to sentiment analysis.

---

## 📌 1. Basic Autoencoder for Image Reconstruction

### ✅ Overview
An autoencoder is a neural network trained to reconstruct its input. In this project, a fully connected autoencoder is trained on the MNIST dataset to compress and decompress images of handwritten digits.

### 🧠 Architecture
- **Encoder**: Input (784) → Dense(32)
- **Decoder**: Dense(32) → Output (784)

### 🛠️ Highlights
- Uses `binary_crossentropy` loss for pixel-wise reconstruction.
- Visualizes original vs reconstructed digits.
- Tests latent space sizes (16, 32, 64) to evaluate reconstruction quality.

### 🔍 Key Insight
Larger latent dimensions preserve more detail, while smaller ones force the model to generalize and compress better.

---

## 🧼 2. Denoising Autoencoder

### ✅ Overview
A denoising autoencoder is trained to reconstruct clean images from noisy inputs. Gaussian noise is added to MNIST digits, and the model learns to remove it.

### 🧠 Architecture
Same as the basic autoencoder.

### 🛠️ Highlights
- Adds Gaussian noise (`np.random.normal`) with `std=0.5`.
- Trains on noisy input but predicts clean output.
- Visualizes Noisy → Denoised → Original images.
- Compares with a regular autoencoder to show noise resilience.

### 🌍 Real-World Use Case
Used in **medical imaging** to enhance scans by removing noise caused by low resolution or equipment limitations.

---

## ✍️ 3. Text Generation with LSTM (Character-Level RNN)

### ✅ Overview
An LSTM-based Recurrent Neural Network is trained to predict the next character in a text sequence using "The Little Prince" as training data.

### 🧠 Architecture
- Input: One-hot encoded characters
- LSTM(128) → Dense(vocab_size, softmax)

### 🛠️ Highlights
- Character-level model captures grammar and syntax.
- Generates text by sampling characters based on predicted probabilities.
- Demonstrates **temperature scaling** to control creativity/randomness.

### 🔥 Temperature Scaling Explained
- Low (0.2): Conservative, repetitive
- Medium (1.0): Balanced
- High (1.5): Creative, but prone to errors

---

## 😊 4. Sentiment Classification using LSTM

### ✅ Overview
An LSTM-based binary classifier trained on the IMDB movie review dataset to detect positive or negative sentiment.

### 🧠 Architecture
- Embedding → LSTM(128) → Dense(1, sigmoid)

### 🛠️ Highlights
- Uses Keras `imdb` dataset (tokenized, padded).
- Evaluates model with:
  - Accuracy
  - Precision
  - Recall
  - F1-score
  - Confusion Matrix

### 🔄 Precision vs Recall in Sentiment Tasks
- **High Recall**: Catch all negative reviews (even with some false positives).
- **High Precision**: Ensure flagged reviews are truly negative.

Ideal tradeoff depends on business use (e.g., customer service, moderation).

---

## 🔧 Requirements

- Python 3.7+
- TensorFlow 2.x
- NumPy, Matplotlib, Seaborn
- scikit-learn (for metrics)

Install via:

```bash
pip install tensorflow numpy matplotlib seaborn scikit-learn
