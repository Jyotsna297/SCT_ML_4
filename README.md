# SCT_ML_4 - Hand Gesture Recognition using Machine Learning

## 📌 Task Description

This project is developed as part of the **SkillCraft Technology Machine Learning Internship (Task 4)**.

The objective is to build a **Hand Gesture Recognition System** that can identify different hand gestures from webcam input using **MediaPipe** for hand landmark detection and a **Random Forest Classifier** for gesture classification.

---

## 🚀 Features

- Real-time hand gesture recognition
- Hand landmark detection using MediaPipe
- Machine Learning classification using Random Forest
- Webcam-based live prediction
- Beginner-friendly implementation

---

## 🛠️ Technologies Used

- Python
- OpenCV
- MediaPipe
- NumPy
- Scikit-learn
- Joblib
- TQDM

---

## 📂 Project Structure

```
SCT_ML_4/
│
├── LeapGestRecog/
│
├── train_model.py
├── gesture_recognition.py
├── gesture_model.pkl
├── requirements.txt
└── README.md
```

---

## 📊 Dataset

Dataset Used:

**LeapGestRecog - Hand Gesture Recognition Dataset**

The model is trained on the following gestures:

- ✋ Palm
- ✊ Fist
- 👍 Thumb
- 👌 OK

---

## ⚙️ Installation

Clone the repository

```bash
git clone https://github.com/your-username/SCT_ML_4.git
```

Move into the project folder

```bash
cd SCT_ML_4
```

Install dependencies

```bash
pip install -r requirements.txt
```

---

## ▶️ Training the Model

Run:

```bash
python train_model.py
```

This will:

- Read the dataset
- Extract hand landmarks using MediaPipe
- Train a Random Forest classifier
- Save the trained model as:

```
gesture_model.pkl
```

---

## 🎥 Running Gesture Recognition

Run:

```bash
python gesture_recognition.py
```

The webcam will open and recognize supported hand gestures in real time.

Press **Q** to quit.

---

## 📈 Model

Algorithm Used:

- Random Forest Classifier

Hand Detection:

- MediaPipe Hands

Input Features:

- 21 Hand Landmarks
- x, y, z coordinates
- Total Features = 63

---

## 📦 Requirements

```
opencv-python
mediapipe
numpy
scikit-learn
joblib
tqdm
```

Install using:

```bash
pip install -r requirements.txt
```

---

## 📸 Output

The system detects and displays the predicted gesture on the webcam feed.

Supported gestures:

- Palm
- Fist
- Thumb
- OK

---

## 🎯 Future Improvements

- Add more gesture classes
- Improve accuracy using Deep Learning (CNN)
- Support dynamic hand gestures
- Deploy as a web application

SkillCraft Technology Machine Learning Internship

Task 4 – Hand Gesture Recognition
