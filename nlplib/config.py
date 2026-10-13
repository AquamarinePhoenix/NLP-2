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
RANDOM_STATE = 17