import speech_recognition as sr
import os
import pickle
import librosa
import numpy as np
import soundfile as sf

# ── Settings ──────────────────────────────────────────────────────────────────
VOICE_DATA_DIR = "voice_data"
PASSPHRASE     = "open the door i am bharath"
SAMPLES_NEEDED = 3   # record 3 samples and average them

# ── Extract MFCC feature vector from audio file ────────────────────────────────
def extract_mfcc(file_path):
    y, sr_rate = librosa.load(file_path, sr=16000, mono=True)
    mfcc       = librosa.feature.mfcc(y=y, sr=sr_rate, n_mfcc=13,
                                       n_fft=512, hop_length=160, win_length=400)
    delta      = librosa.feature.delta(mfcc)
    delta2     = librosa.feature.delta(mfcc, order=2)
    features   = np.vstack([mfcc, delta, delta2])   # shape (39, T)
    return np.mean(features, axis=1)                 # shape (39,)

# ── Record one voice sample ────────────────────────────────────────────────────
def record_sample(sample_number, save_path):
    recogniser = sr.Recognizer()

    with sr.Microphone(sample_rate=16000) as source:
        print(f"\nSample {sample_number}/{SAMPLES_NEEDED}")
        print(f'Say: "{PASSPHRASE}"')
        print("Recording in 3 seconds... get ready.")

        recogniser.adjust_for_ambient_noise(source, duration=1)
        print("RECORDING NOW...")

        try:
            audio = recogniser.listen(source, timeout=5, phrase_time_limit=6)
            print("Got it. Saving...")

            # Save as WAV
            with open(save_path, "wb") as f:
                f.write(audio.get_wav_data())

            print(f"Saved: {save_path}")
            return True

        except sr.WaitTimeoutError:
            print("Timeout — no speech detected. Try again.")
            return False

# ── Enrol user voice ───────────────────────────────────────────────────────────
def enrol_voice(user_name):
    save_dir = os.path.join(VOICE_DATA_DIR, user_name)
    os.makedirs(save_dir, exist_ok=True)

    print(f"\nVoice enrolment for: {user_name}")
    print(f'You will say "{PASSPHRASE}" three times.')
    print("Speak clearly and at normal volume.\n")

    vectors = []
    sample  = 1

    while sample <= SAMPLES_NEEDED:
        wav_path = os.path.join(save_dir, f"sample_{sample}.wav")
        success  = record_sample(sample, wav_path)

        if success:
            try:
                vec = extract_mfcc(wav_path)
                vectors.append(vec)
                sample += 1
            except Exception as e:
                print(f"Could not process audio: {e}. Try again.")
        else:
            print("Please try again.")

    # Average the three vectors into one template
    template = np.mean(vectors, axis=0)

    # Save template
    template_path = os.path.join(save_dir, "template.pkl")
    with open(template_path, "wb") as f:
        pickle.dump(template, f)

    print(f"\nVoice template saved for {user_name}.")
    print("Voice enrolment complete.")

# ── Run ───────────────────────────────────────────────────────────────────────
if __name__ == "__main__":
    name = input("Enter your name: ").strip()
    if name:
        enrol_voice(name)
    else:
        print("No name entered.")