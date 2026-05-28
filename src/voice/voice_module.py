import numpy as np
import librosa
import sounddevice as sd
import scipy.io.wavfile as wav
import os
from scipy.signal import butter, filtfilt

# ── BANDPASS FILTER ────────────────────────────────────────
def bandpass_filter(audio, sr, low=300, high=3400):
    b, a = butter(4, [low / (sr / 2), high / (sr / 2)], btype='band')
    return filtfilt(b, a, audio)

# ── MFCC EXTRACTION ────────────────────────────────────────
def extract_mfcc(audio_path, n_mfcc=13):
    y, sr = librosa.load(audio_path, sr=16000)
    y = bandpass_filter(y, sr)
    mfcc = librosa.feature.mfcc(y=y, sr=sr, n_mfcc=n_mfcc)
    return np.mean(mfcc.T, axis=0)

# ── RECORD VOICE ───────────────────────────────────────────
def record_voice(duration=3, sr=16000, save_path='temp_voice.wav'):
    print(f"  Recording for {duration} seconds... speak now!")
    audio = sd.rec(int(duration * sr), samplerate=sr,
                   channels=1, dtype='float32')
    sd.wait()
    audio = (audio.flatten() * 32767).astype(np.int16)
    wav.write(save_path, sr, audio)
    print("  Recording complete.")
    return save_path

# ── ENROL USER ─────────────────────────────────────────────
def enrol_voice(user_id, save_dir='voice_templates'):
    os.makedirs(save_dir, exist_ok=True)
    print(f"Enrolling voice for user {user_id}...")
    print("Say your passphrase 3 times:")
    templates = []
    for i in range(3):
        print(f"  Attempt {i+1}/3:")
        path = record_voice(save_path=f'enrol_{i}.wav')
        mfcc = extract_mfcc(path)
        templates.append(mfcc)
    template = np.mean(templates, axis=0)
    np.save(os.path.join(save_dir, f'{user_id}.npy'), template)
    print(f"Voice template saved for user {user_id}.")
    return template

# ── VERIFY USER ────────────────────────────────────────────
def verify_voice(user_id, save_dir='voice_templates'):
    template_path = os.path.join(save_dir, f'{user_id}.npy')
    if not os.path.exists(template_path):
        print("No voice template found. Please enrol first.")
        return 0.0
    template = np.load(template_path)
    print("Speak your passphrase now:")
    path = record_voice(save_path='verify_voice.wav')
    mfcc = extract_mfcc(path)
    score = np.dot(template, mfcc) / (
        np.linalg.norm(template) * np.linalg.norm(mfcc))
    score = float(np.clip(score, 0, 1))
    print(f"  Voice score: {score:.4f}")
    return score

# ── SCORE NORMALIZATION ────────────────────────────────────
def normalize_voice_score(score):
    return round(float(np.clip(score, 0.0, 1.0)), 4)
