import cv2
import pickle

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/face_model.yml"
LABELS_PATH   = "models/face_labels.pkl"
CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
THRESHOLD     = 100  # lower = stricter. If confidence > this, shows "Unknown"

# ── Load model and labels ──────────────────────────────────────────────────────
recogniser = cv2.face.LBPHFaceRecognizer_create()
recogniser.read(MODEL_PATH)

with open(LABELS_PATH, "rb") as f:
    label_map = pickle.load(f)

face_cascade = cv2.CascadeClassifier(CASCADE_PATH)

# ── Run live recognition ───────────────────────────────────────────────────────
print("Starting face recognition. Press Q to quit.")

cap = cv2.VideoCapture(0)

while True:
    ret, frame = cap.read()
    if not ret:
        break

    grey = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
    faces = face_cascade.detectMultiScale(grey, 1.1, 5, minSize=(80, 80))

    for (x, y, w, h) in faces:
        face_img = grey[y:y+h, x:x+w]
        face_img = cv2.resize(face_img, (200, 200))
        face_img = cv2.equalizeHist(face_img)

        label, confidence = recogniser.predict(face_img)

        if confidence < THRESHOLD:
            name    = label_map[label]
            display = f"{name}  ({confidence:.1f})"
            color   = (47, 139, 58)    # green = recognised
        else:
            display = f"Unknown  ({confidence:.1f})"
            color   = (47, 58, 139)    # red = not recognised

        cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
        cv2.putText(frame, display, (x, y - 10),
                    cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

    cv2.putText(frame, "Press Q to quit",
                (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
    cv2.imshow("Face Recognition Test", frame)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
cv2.destroyAllWindows()
print("Done.")