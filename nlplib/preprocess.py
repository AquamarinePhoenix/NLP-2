import re
import pandas as pd
from nlplib.config import WORDS, DATASET_DIR, AUDIO_FORMAT

def return_dimAudio() -> pd.DataFrame:
    dimAudio = []

    for word in WORDS:
        word_dir = DATASET_DIR / word

        for file_path in word_dir.glob("*" + AUDIO_FORMAT):

            file = re.match(r"(.+)_nohash_\d+\.wav$", file_path.name)
            dimAudio.append({
                "word": word,
                "speaker_id": file.group(1)
            })
            
    return pd.DataFrame(dimAudio)