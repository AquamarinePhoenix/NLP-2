import pandas as pd
from nlplib.write import write_file
from nlplib.config import WORDS

def output_audio_partition(dimAudio: pd.DataFrame, train_dimAudio: pd.DataFrame, test_dimAudio: pd.DataFrame) -> None:
    write_file(f"Train/test ratio: {len(train_dimAudio)/len(dimAudio):.1%}:{len(test_dimAudio)/len(dimAudio):.1%}")
    write_file(f"Train, test speakers: {len(set(train_dimAudio["speaker_id"]))}, {len(set(test_dimAudio["speaker_id"]))}")
    write_file("Training samples per word:")
    write_file(train_dimAudio["word"].value_counts().reindex(WORDS))
    write_file("Test samples per word:")
    write_file(test_dimAudio["word"].value_counts().reindex(WORDS))