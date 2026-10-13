#%%

from sklearn.model_selection import GroupShuffleSplit
import matplotlib.pyplot as plt
from nlplib.config import WORDS, RANDOM_STATE, TEST_SIZE
from nlplib.write import clear_file
from nlplib.preprocess import return_dimAudio
from nlplib.output import output_audio_partition
from nlplib.plot import histogram_plot

def main():
    clear_file()
    dimAudio = return_dimAudio()

    splitter = GroupShuffleSplit(n_splits = 25, test_size = TEST_SIZE, random_state = RANDOM_STATE)

    for train_idx, test_idx in splitter.split(dimAudio, y = dimAudio["word"], groups = dimAudio["speaker_id"]):
        train_dimAudio = dimAudio.iloc[train_idx].copy()
        test_dimAudio = dimAudio.iloc[test_idx].copy()

        if (
            set(train_dimAudio["word"]) == set(WORDS)
            and set(test_dimAudio["word"]) == set(WORDS)
        ):
            break

    output_audio_partition(dimAudio, train_dimAudio, test_dimAudio)
    histogram_plot(train_dimAudio["word"], bins = 10, title = "Training set distribution")
    histogram_plot(test_dimAudio["word"], bins = 10, title = "Test set distribution")

if __name__ == "__main__":
    main()