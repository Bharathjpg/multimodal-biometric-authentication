import cv2
import numpy as np
import os
from sklearn.model_selection import KFold

# ── PREPROCESSING ──────────────────────────────────────────
def preprocess_face(img_path, apply_eq=True, apply_sharpen=True):
    img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
    img = cv2.resize(img, (92, 112))
    if apply_eq:
        img = cv2.equalizeHist(img)
    if apply_sharpen:
        blurred = cv2.GaussianBlur(img, (0, 0), 3)
        img = cv2.addWeighted(img, 1.5, blurred, -0.5, 0)
    return img

# ── LOAD DATASET ───────────────────────────────────────────
def load_att_dataset(dataset_path):
    images, labels = [], []
    for subject_id, subject_dir in enumerate(sorted(os.listdir(dataset_path))):
        subject_path = os.path.join(dataset_path, subject_dir)
        if not os.path.isdir(subject_path):
            continue
        for img_file in os.listdir(subject_path):
            img_path = os.path.join(subject_path, img_file)
            img = preprocess_face(img_path)
            if img is not None:
                images.append(img)
                labels.append(subject_id)
    return images, labels

# ── TRAIN AND EVALUATE ─────────────────────────────────────
def cross_validate(images, labels, algorithm='lbph', n_splits=5):
    images = np.array(images)
    labels = np.array(labels)
    kf = KFold(n_splits=n_splits, shuffle=True, random_state=42)

    accuracies = []
    for fold, (train_idx, test_idx) in enumerate(kf.split(images)):
        X_train, X_test = images[train_idx], images[test_idx]
        y_train, y_test = labels[train_idx], labels[test_idx]

        if algorithm == 'lbph':
            model = cv2.face.LBPHFaceRecognizer_create(
                radius=1, neighbors=8, grid_x=8, grid_y=8)
        elif algorithm == 'eigenfaces':
            model = cv2.face.EigenFaceRecognizer_create(num_components=40)
        elif algorithm == 'fisherfaces':
            model = cv2.face.FisherFaceRecognizer_create(num_components=39)

        model.train(list(X_train), y_train)

        correct = 0
        for img, true_label in zip(X_test, y_test):
            predicted_label, _ = model.predict(img)
            if predicted_label == true_label:
                correct += 1

        accuracy = correct / len(y_test) * 100
        accuracies.append(accuracy)
        print(f"  Fold {fold+1}: {accuracy:.1f}%")

    mean_acc = np.mean(accuracies)
    std_acc = np.std(accuracies)
    print(f"\n  Mean Accuracy: {mean_acc:.1f}% | Std: {std_acc:.1f}%")
    return mean_acc, std_acc

# ── SCORE NORMALIZATION ────────────────────────────────────
def normalize_score(confidence, max_conf=200.0):
    # Lower confidence value = better match in OpenCV
    score = 1.0 - min(confidence / max_conf, 1.0)
    return round(score, 4)
