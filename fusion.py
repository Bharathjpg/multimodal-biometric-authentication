import cv2
import pickle
import numpy as np
import os
import speech_recognition as sr
import librosa
from kba_module import authenticate_kba

# ── Settings ──────────────────────────────────────────────────────────────────
MODEL_PATH    = "models/face_model.yml"
LABELS_PATH   = "models/face_labels.pkl"
CASCADE_PATH  = cv2.data.haarcascades + "haarcascade_frontalface_default.xml"
VOICE_DATA_DIR = "voice_data"
PASSPHRASE    = "open the door i am bharath"

FACE_THRESHOLD  = 100.0
VOICE_THRESHOLD = 40.0
DECISION_THETA  = 0.50   # final fusion threshold

# Module weights (derived from accuracy)
W_FACE  = 0.332
W_VOICE = 0.321
W_KBA   = 0.347

# ── Face recognition ──────────────────────────────────────────────────────────
def recognise_face(user_name):
    recogniser = cv2.face.LBPHFaceRecognizer_create()
    recogniser.read(MODEL_PATH)

    with open(LABELS_PATH, "rb") as f:
        label_map = pickle.load(f)

    face_cascade = cv2.CascadeClassifier(CASCADE_PATH)
    cap = cv2.VideoCapture(0)

    print("\n--- MODULE 1: FACE RECOGNITION ---")
    print("Look at the camera. Press SPACE when your face is in the green box.")

    score = 0.0
    while True:
        ret, frame = cap.read()
        if not ret:
            break

        grey  = cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)
        faces = face_cascade.detectMultiScale(grey, 1.1, 5, minSize=(80, 80))

        for (x, y, w, h) in faces:
            face_img = grey[y:y+h, x:x+w]
            face_img = cv2.resize(face_img, (200, 200))
            face_img = cv2.equalizeHist(face_img)

            label, confidence = recogniser.predict(face_img)

            if confidence < FACE_THRESHOLD:
                name    = label_map[label]
                display = f"{name} ({confidence:.1f})"
                color   = (47, 139, 58)
            else:
                display = f"Unknown ({confidence:.1f})"
                color   = (47, 58, 139)

            cv2.rectangle(frame, (x, y), (x+w, y+h), color, 2)
            cv2.putText(frame, display, (x, y-10),
                        cv2.FONT_HERSHEY_SIMPLEX, 0.8, color, 2)

        cv2.putText(frame, "Press SPACE to capture | Q to quit",
                    (10, 30), cv2.FONT_HERSHEY_SIMPLEX, 0.6, (100, 100, 100), 1)
        cv2.imshow("Face Recognition", frame)

        key = cv2.waitKey(1) & 0xFF

        if key == ord(' '):
            if len(faces) == 1:
                face_img = grey[faces[0][1]:faces[0][1]+faces[0][3],
                                faces[0][0]:faces[0][0]+faces[0][2]]
                face_img = cv2.resize(face_img, (200, 200))
                face_img = cv2.equalizeHist(face_img)
                label, confidence = recogniser.predict(face_img)

                raw_score = max(0.0, 1.0 - confidence / FACE_THRESHOLD)
                score     = float(np.clip(raw_score, 0.0, 1.0))

                name = label_map.get(label, "Unknown")
                print(f"Face captured: {name}  confidence={confidence:.1f}  score={score:.2f}")
                break
            else:
                print("No clear face detected. Try again.")

        elif key == ord('q'):
            break

    cap.release()
    cv2.destroyAllWindows()
    return score

# ── Voice authentication ───────────────────────────────────────────────────────
def extract_mfcc(file_path):
    y, sr_rate = librosa.load(file_path, sr=16000, mono=True)
    mfcc       = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13,
                                       n_fft=512, hop_length=160, win_length=400)
    delta      = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)
    return np.mean(np.vstack([mfcc, delta, delta2]), axis=1)

def recognise_voice(user_name):
    template_path = os.path.join(VOICE_DATA_DIR, user_name, "template.pkl")

    if not os.path.exists(template_path):
        print("No voice template found.")
        return 0.0

    with open(template_path, "rb") as f:
        template = pickle.load(f)

    recogniser = sr.Recognizer()
    print("\n--- MODULE 2: VOICE AUTHENTICATION ---")
    print(f'Say: "{PASSPHRASE}"')

    with sr.Microphone(sample_rate=16000) as source:
        recogniser.adjust_for_ambient_noise(source, duration=1)
        print("RECORDING NOW...")
        try:
            audio    = recogniser.listen(source, timeout=5, phrase_time_limit=6)
            wav_path = os.path.join(VOICE_DATA_DIR, "live.wav")
            with open(wav_path, "wb") as f:
                f.write(audio.get_wav_data())
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return 0.0

    live_vec = extract_mfcc(wav_path)
    distance = np.linalg.norm(live_vec - template)
    score    = float(np.clip(1.0 - distance / VOICE_THRESHOLD, 0.0, 1.0))

    print(f"Voice distance={distance:.2f}  score={score:.2f}")
    return score

# ── WSR Fusion and decision ────────────────────────────────────────────────────
def fuse_and_decide(s_face, s_voice, s_kba):
    s_fused = W_FACE * s_face + W_VOICE * s_voice + W_KBA * s_kba

    print("\n--- FUSION RESULT ---")
    print(f"Face  score : {s_face:.2f}  (weight {W_FACE})")
    print(f"Voice score : {s_voice:.2f}  (weight {W_VOICE})")
    print(f"KBA   score : {s_kba:.2f}  (weight {W_KBA})")
    print(f"Fused score : {s_fused:.2f}  (threshold {DECISION_THETA})")

    if s_fused >= DECISION_THETA:
        print("\n*** ACCESS GRANTED — UNLOCK ***")
    else:
        print("\n*** ACCESS DENIED — LOCK ***")

    return s_fused

# ── Main ──────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("=== LOCKING/UNLOCKING DECISION SYSTEM ===")
    user_name = input("Enter your name: ").strip()

    s_face  = recognise_face(user_name)
    s_voice = recognise_voice(user_name)
    s_kba   = authenticate_kba(user_name)

    fuse_and_decide(s_face, s_voice, s_kba)