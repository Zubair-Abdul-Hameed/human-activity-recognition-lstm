# Human Activity Recognition using LSTM (UCI HAR Dataset)

This project implements a deep learning model using **LSTM (Long Short-Term Memory networks)** to classify human activities from smartphone sensor data (accelerometer and gyroscope signals).

---

## 📊 Dataset

- **UCI Human Activity Recognition Dataset**
- Sensor signals:
  - Body acceleration (X, Y, Z)
  - Gyroscope (X, Y, Z)
  - Total acceleration (X, Y, Z)
- Window size: 128 timesteps
- Number of features: 9
- Number of classes: 6

---

## 🧠 Model Architecture

- Stacked **LSTM (2 layers)**
- Hidden size: 128
- Dropout: 0.3
- Fully connected output layer
- Input shape: `(batch_size, 128, 9)`
- Output: 6 activity classes

---

## 🚶 Activities

The model classifies the following human activities:

- Walking
- Walking Upstairs
- Walking Downstairs
- Sitting
- Standing
- Laying

---

## ⚙️ Tech Stack

- Python
- PyTorch
- NumPy
- Scikit-learn

---

## 🧹 Preprocessing

- Standardization (z-score normalization)
- Computed only on training data
- Applied to both train and test sets
- Ensures:
  - Mean ≈ 0
  - Std ≈ 1

---

## 🏋️ Training Setup

- Loss function: CrossEntropyLoss
- Optimizer: Adam
- Learning rate scheduler: ReduceLROnPlateau
- Early stopping (patience-based)
- Gradient clipping (max norm = 1.0)
- Batch size: 64

---

## 📈 Results

- Test Accuracy: **~89.99%**
- Best validation accuracy: ~90%+
- Strong performance on most classes, with minor confusion between similar activities (e.g., sitting vs standing)

---

## 📊 Evaluation Metrics

- Classification Report (Precision, Recall, F1-score)
- Confusion Matrix
- Overall Accuracy

---

## 📁 Project Structure
src/
├── data_loader.py
├── dataset.py
├── model.py
├── train.py
├── evaluate.py
├── preprocessing.py
├── plot_results.py
main.py



---

## 🚀 How to Run

- pip install -r requirements.txt
- python main.py

##📌 Notes
- Best model is saved during training (models/best_model.pt)
- Training uses early stopping to prevent overfitting
- All preprocessing is done using training statistics only (no data leakage)
##📌 Future Improvements
- Hyperparameter tuning (hidden size, layers, dropout)
- Attention-based LSTM
- Real-time inference on mobile sensor data