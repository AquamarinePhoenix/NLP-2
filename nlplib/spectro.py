import librosa
import numpy as np
from nlplib.config import SAMPLE_RATE, NUM_SAMPLES, FFT_SIZE, FRAME_LENGTH, HOP_LENGTH, NUM_MELS

def extract_features_from_audio(audio: np.ndarray) -> np.ndarray:
    if len(audio) < NUM_SAMPLES:
        audio = np.pad(audio, (0, NUM_SAMPLES - len(audio)), mode = "constant")
    else:
        audio = audio[:NUM_SAMPLES]

    mel_spec = librosa.feature.melspectrogram(
        y=audio,
        sr = SAMPLE_RATE,
        n_fft = FFT_SIZE,
        win_length = FRAME_LENGTH,
        hop_length = HOP_LENGTH,
        window = "hann",
        n_mels = NUM_MELS,
        power = 2.0
    )

    log_mel_spec = librosa.power_to_db(mel_spec, ref = 1.0)

    return log_mel_spec


def extract_features(filepath) -> np.ndarray:
    audio, _ = librosa.load(filepath, sr = SAMPLE_RATE, mono = True)
    return extract_features_from_audio(audio)