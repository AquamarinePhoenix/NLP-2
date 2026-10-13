from pathlib import Path

import librosa
import numpy as np
import pandas as pd

from nlplib.config import DATASET_DIR, SAMPLE_RATE, NUM_SAMPLES, WORDS
from nlplib.spectro import extract_features_from_audio
from nlplib.write import write_file


def _rms(audio: np.ndarray) -> float:
    return float(np.sqrt(np.mean(np.square(audio))) + 1e-10)


def _scale_to_snr(signal: np.ndarray, noise: np.ndarray, snr_db: float) -> np.ndarray:
    target_noise_rms = _rms(signal) / (10 ** (snr_db / 20))
    return noise * (target_noise_rms / _rms(noise))


def _fit_length(audio: np.ndarray, length: int) -> np.ndarray:
    if len(audio) < length:
        repeats = int(np.ceil(length / len(audio)))
        audio = np.tile(audio, repeats)
    return audio[:length]


def _load_audio(filepath) -> np.ndarray:
    audio, _ = librosa.load(filepath, sr = SAMPLE_RATE, mono = True)
    return _fit_length(audio, NUM_SAMPLES)


def _background_files() -> list[Path]:
    return sorted((DATASET_DIR / "_background_noise_").glob("*.wav"))


def _noisy_audio(
    audio: np.ndarray,
    noise: np.ndarray,
    snr_db: float,
) -> np.ndarray:
    noise = _fit_length(noise, len(audio))
    return audio + _scale_to_snr(audio, noise, snr_db)


def evaluate_noise_conditions(
    model,
    test_dimAudio: pd.DataFrame,
    mean: float,
    std: float,
    snr_levels: tuple[int, ...] = (30, 20, 10, 0),
) -> None:
    label_to_id = {word: i for i, word in enumerate(WORDS)}
    labels = test_dimAudio["word"].map(label_to_id).to_numpy()
    background_files = _background_files()

    if not background_files:
        raise FileNotFoundError(
            f"No background recordings found in {DATASET_DIR / '_background_noise_'}"
        )

    conditions: list[tuple[str, int | None]] = [("Clean", None)]
    conditions.extend(("White noise", snr_db) for snr_db in snr_levels)
    conditions.extend(("Background noise", snr_db) for snr_db in snr_levels)

    for condition, snr_db in conditions:
        features = []
        for index, filepath in enumerate(test_dimAudio["file_path"]):
            audio = _load_audio(filepath)
            if condition == "White noise":
                noise = np.random.default_rng(index).normal(0, 1, len(audio))
                audio = _noisy_audio(audio, noise, snr_db)
            elif condition == "Background noise":
                noise = _load_audio(background_files[index % len(background_files)])
                audio = _noisy_audio(audio, noise, snr_db)
            features.append(extract_features_from_audio(audio))

        inputs = np.stack(features)[..., np.newaxis].astype("float32")
        inputs = (inputs - mean) / std
        loss, accuracy = model.evaluate(inputs, labels, verbose = 0)
        suffix = "" if snr_db is None else f" ({snr_db} dB)"
        write_file(f"{condition}{suffix} - loss: {loss:.4f}, accuracy: {accuracy:.4f}")
