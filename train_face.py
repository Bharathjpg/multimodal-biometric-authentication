import cv2
import os
import numpy as np
import pickle

# ── Settings ──────────────────────────────────────────────────────────────────
FACE_DATA_DIR = "face_data"
MODEL_PATH    = "models/face_model.yml"
LABELS_PATH   = "models/face_labels.pkl"

# ── Load all face images ───────────────────────────────────────────────────────
def load_face_data():
    images = []
    labels = []
    label_map = {}   # name -> number
    current_label = 0

    for person_name in sorted(os.listdir(FACE_DATA_DIR)):
        person_dir = os.path.join(FACE_DATA_DIR, person_name)
        if not os.path.isdir(person_dir):
            continue

        label_map[current_label] = person_name
        print(f"  Loading: {person_name} (label {current_label})")

        for img_file in sorted(os.listdir(person_dir)):
            img_path = os.path.join(person_dir, img_file)
            img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
            if img is not None:
                images.append(img)
                labels.append(current_label)

        current_label += 1

    return images, labels, label_map

# ── Train ──────────────────────────────────────────────────────────────────────
def train():
    print("\nLoading face data...")
    images, labels, label_map = load_face_data()

    if len(images) == 0:
        print("ERROR: No images found in face_data folder.")
        return

    print(f"\nTraining LBPH on {len(images)} images...")

    recogniser = cv2.face.LBPHFaceRecognizer_create(
        radius=3, neighbors=8, grid_x=4, grid_y=4
    )
    recogniser.train(images, np.array(labels))

    # Save model
    os.makedirs("models", exist_ok=True)
    recogniser.save(MODEL_PATH)

    # Save label map
    with open(LABELS_PATH, "wb") as f:
        pickle.dump(label_map, f)

    print(f"\nModel saved to: {MODEL_PATH}")
    print(f"Labels saved to: {LABELS_PATH}")
    print(f"\nEnrolled users: {list(label_map.values())}")
    print("\nTraining complete.")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    train()