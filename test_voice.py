import speech_recognition as sr
import librosa
import numpy as np
import pickle
import os

# ── Settings ──────────────────────────────────────────────────────────────────
VOICE_DATA_DIR = "voice_data"
PASSPHRASE     = "open the door i am bharath"
THRESHOLD      = 40.0   # lower = stricter match

# ── Extract MFCC ──────────────────────────────────────────────────────────────
def extract_mfcc(file_path):
    y, sr_rate = librosa.load(file_path, sr=16000, mono=True)
    mfcc       = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13,
                                       n_fft=512, hop_length=160, win_length=400)
    delta      = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)
    features   = np.vstack([mfcc, delta, delta2])
    return np.mean(features, axis=1)

# ── Record live sample ────────────────────────────────────────────────────────
def record_live(save_path="voice_data/live_test.wav"):
    recogniser = sr.Recognizer()
    with sr.Microphone(sample_rate=16000) as source:
        print(f'\nSay: "{PASSPHRASE}"')
        recogniser.adjust_for_ambient_noise(source, duration=1)
        print("RECORDING NOW...")
        try:
            audio = recogniser.listen(source, timeout=5, phrase_time_limit=6)
            with open(save_path, "wb") as f:
                f.write(audio.get_wav_data())
            return save_path
        except sr.WaitTimeoutError:
            print("No speech detected.")
            return None

# ── Compute normalised score ───────────────────────────────────────────────────
def voice_score(live_vec, template_vec, threshold=THRESHOLD):
    distance = np.linalg.norm(live_vec - template_vec)
    score    = max(0.0, 1.0 - distance / threshold)
    return score, distance

# ── Test ──────────────────────────────────────────────────────────────────────
def test_voice(user_name):
    template_path = os.path.join(VOICE_DATA_DIR, user_name, "template.pkl")

    if not os.path.exists(template_path):
        print(f"No voice template found for {user_name}.")
        return

    with open(template_path, "rb") as f:
        template = pickle.load(f)

    print(f"\nTesting voice for: {user_name}")
    live_path = record_live()

    if live_path is None:
        print("Recording failed.")
        return

    live_vec       = extract_mfcc(live_path)
    score, distance = voice_score(live_vec, template)

    print(f"\nDistance : {distance:.2f}")
    print(f"Score    : {score:.2f}  (0=no match, 1=perfect match)")

    if score > 0.3:
        print(f"Result   : VOICE MATCH - Welcome {user_name}")
    else:
        print(f"Result   : VOICE NOT MATCHED")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    name = input("Enter your name to test: ").strip()
    if name:
        test_voice(name)
    else:
        print("No name entered.")