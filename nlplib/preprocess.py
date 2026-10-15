import re
import pandas as pd
from sklearn.model_selection import GroupShuffleSplit
from nlplib.config import WORDS, DATASET_DIR, AUDIO_FORMAT, TEST_SIZE, SEED

def return_dimAudio() -> pd.DataFrame:
    dimAudio = []

    for word in WORDS:
        word_dir = DATASET_DIR / word

        for file_path in word_dir.glob("*" + AUDIO_FORMAT):

            file = re.match(r"(.+)_nohash_\d+\.wav$", file_path.name)
            dimAudio.append({
                "file_path": file_path,
                "word": word,
                "speaker_id": file.group(1)
            })
            
    return pd.DataFrame(dimAudio)

def split_train_test(dimAudio: pd.DataFrame,):
    splitter = GroupShuffleSplit(n_splits = 25, test_size = TEST_SIZE, random_state = SEED)

    for train_idx, test_idx in splitter.split(dimAudio, y = dimAudio["word"], groups = dimAudio["speaker_id"]):
        train_dimAudio = dimAudio.iloc[train_idx].copy()
        test_dimAudio = dimAudio.iloc[test_idx].copy()

        if (
            set(train_dimAudio["word"]) == set(WORDS)
            and set(test_dimAudio["word"]) == set(WORDS)
        ):
            train_speakers = set(train_dimAudio["speaker_id"])
            test_speakers = set(test_dimAudio["speaker_id"])
            if train_speakers.isdisjoint(test_speakers):
                return train_dimAudio, test_dimAudio