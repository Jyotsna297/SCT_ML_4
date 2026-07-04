import os
import cv2
import joblib
import mediapipe as mp
import numpy as np

from tqdm import tqdm
from sklearn.model_selection import train_test_split
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import accuracy_score

# -----------------------------
# DATASET LOCATION
# -----------------------------
DATASET_PATH = "LeapGestRecog"

# Gestures to use
GESTURES = {
    "01_palm": "Palm",
    "03_fist": "Fist",
    "05_thumb": "Thumb",
    "07_ok": "OK"
}

# -----------------------------
# MEDIAPIPE
# -----------------------------
mp_hands = mp.solutions.hands

hands = mp_hands.Hands(
    static_image_mode=True,
    max_num_hands=1,
    min_detection_confidence=0.5
)

X = []
Y = []

print("Loading Dataset...")

# -----------------------------
# READ ALL SUBJECTS
# -----------------------------
subjects = sorted(os.listdir(DATASET_PATH))

for subject in subjects:

    subject_path = os.path.join(DATASET_PATH, subject)

    if not os.path.isdir(subject_path):
        continue

    print("Reading Subject:", subject)

    for folder in os.listdir(subject_path):

        if folder not in GESTURES:
            continue

        label = GESTURES[folder]

        folder_path = os.path.join(subject_path, folder)

        images = os.listdir(folder_path)

        for image_name in tqdm(images):

            image_path = os.path.join(folder_path, image_name)

            image = cv2.imread(image_path)

            if image is None:
                continue

            rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)

            result = hands.process(rgb)

            if result.multi_hand_landmarks:

                hand = result.multi_hand_landmarks[0]

                features = []

                for lm in hand.landmark:

                    features.append(lm.x)
                    features.append(lm.y)
                    features.append(lm.z)

                X.append(features)
                Y.append(label)

print()

print("Images Processed :", len(X))

X = np.array(X)
Y = np.array(Y)

print("Dataset Shape :", X.shape)
# -----------------------------
# TRAIN TEST SPLIT
# -----------------------------
print("\nSplitting Dataset...")

X_train, X_test, Y_train, Y_test = train_test_split(
    X,
    Y,
    test_size=0.2,
    random_state=42,
    stratify=Y
)

print("Training Samples :", len(X_train))
print("Testing Samples  :", len(X_test))

# -----------------------------
# TRAIN MODEL
# -----------------------------
print("\nTraining Random Forest Model...")

model = RandomForestClassifier(
    n_estimators=200,
    random_state=42
)

model.fit(X_train, Y_train)

# -----------------------------
# TEST MODEL
# -----------------------------
print("\nTesting Model...")

predictions = model.predict(X_test)

accuracy = accuracy_score(Y_test, predictions)

print("\n==============================")
print("MODEL ACCURACY")
print("==============================")
print(f"Accuracy : {accuracy*100:.2f}%")

# -----------------------------
# SAVE MODEL
# -----------------------------
joblib.dump(model, "gesture_model.pkl")

print("\nModel saved successfully!")
print("File created : gesture_model.pkl")

hands.close()

print("\nTraining Completed Successfully!")