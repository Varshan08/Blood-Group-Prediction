import os
import cv2
import numpy as np
from sklearn.svm import LinearSVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from joblib import dump

dataset_path = "dataset"

data = []
labels = []

for label in os.listdir(dataset_path):
    folder = os.path.join(dataset_path, label)
    if not os.path.isdir(folder):
        continue

    for img_name in os.listdir(folder):
        img_path = os.path.join(folder, img_name)

        img = cv2.imread(img_path, 0)
        if img is None:
            continue

        img = cv2.resize(img, (100, 100))
        data.append(img.flatten())
        labels.append(label)

data = np.array(data)
labels = np.array(labels)

X_train, X_test, y_train, y_test = train_test_split(
    data, labels, test_size=0.2, random_state=42
)

# ⚡ FAST + SMALL
svm = LinearSVC()
svm.fit(X_train, y_train)

rf = RandomForestClassifier(
    n_estimators=50,
    max_depth=10,
    n_jobs=-1
)
rf.fit(X_train, y_train)

dump(svm, "svm_model.joblib")
dump(rf, "rf_model.joblib")

print("✅ ML Models saved!")