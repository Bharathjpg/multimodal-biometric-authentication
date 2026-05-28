import librosa
import numpy as np
import pickle
import os

VOICE_DATA_DIR = "voice_data"

def extract_mfcc(file_path):
    y, sr_rate = librosa.load(file_path, sr=16000, mono=True)
    mfcc       = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13,
                                       n_fft=512, hop_length=160, win_length=400)
    delta      = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)
    features   = np.vstack([mfcc, delta, delta2])
    return np.mean(features, axis=1)

def check(user_name):
    folder        = os.path.join(VOICE_DATA_DIR, user_name)
    template_path = os.path.join(folder, "template.pkl")

    with open(template_path, "rb") as f:
        template = pickle.load(f)

    print(f"\nChecking distances for: {user_name}")
    print("(Lower = closer to your template)\n")

    for i in range(1, 4):
        wav_path = os.path.join(folder, f"sample_{i}.wav")
        if os.path.exists(wav_path):
            vec      = extract_mfcc(wav_path)
            distance = np.linalg.norm(vec - template)
            print(f"  Sample {i} distance: {distance:.2f}")

if __name__ == "__main__":
    name = input("Enter your name: ").strip()
    check(name)