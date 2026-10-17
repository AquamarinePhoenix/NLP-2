import librosa
import numpy as np
from nlplib.config import SAMPLE_RATE, NUM_SAMPLES, FFT_SIZE, FRAME_LENGTH, HOP_LENGTH, NUM_MELS
from nlplib.plot import mel_spectrogram_plot
from nlplib.output import output_audio_features

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

def get_features(list: list[dict[str, str]], dataset_name: str = "dataset", show = False):
    list["features"] = list["file_path"].apply(extract_features)
    output_audio_features(list)

    mel_spectrogram_plot(
        list["features"].iloc[0],
        title = (
            f'Log-mel spectrogram for {dataset_name} set'
            f' using word "{list["word"].iloc[0]}"'
        ),
        show = show
    )
    
    return list