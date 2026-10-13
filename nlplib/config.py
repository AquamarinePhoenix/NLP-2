from pathlib import Path

WORDS = [
    "zero", "one", "two", "three", "four",
    "five", "six", "seven", "eight", "nine"
]

DATASET_DIR = Path("data")
OUTPUT_DIR = Path("output")
THEME_PATH = Path(__file__).resolve().parent / "themes" / "powerbi_theme.json"

AUDIO_FORMAT = ".wav"

TEST_SIZE = 0.2
SEED = 17
SAMPLE_RATE = 16000
DURATION = 1.0
NUM_SAMPLES = int(SAMPLE_RATE * DURATION)
FRAME_LENGTH = 400
HOP_LENGTH = 160
FFT_SIZE = 512
NUM_MELS = 40