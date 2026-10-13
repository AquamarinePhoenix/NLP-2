#%%

from nlplib.write import clear_file, write_file
from nlplib.preprocess import return_dimAudio, split_train_test
from nlplib.output import output_audio_partition, output_audio_features
from nlplib.plot import histogram_plot
from nlplib.spectro import extract_features
from nlplib.CNN import run_CNN
from nlplib.noise import evaluate_noise_conditions
#%%
def main():
    #%%
    clear_file()
    write_file("1. Dataset Preparation")
    dimAudio = return_dimAudio()
    train_dimAudio, test_dimAudio = split_train_test(dimAudio)

    output_audio_partition(dimAudio, train_dimAudio, test_dimAudio)
    histogram_plot(train_dimAudio["word"], bins = 10, title = "Training set distribution", show = False)
    histogram_plot(test_dimAudio["word"], bins = 10, title = "Test set distribution", show = False)
#%%
    write_file("2. Feature Extraction")
    train_dimAudio["features"] = train_dimAudio["file_path"].apply(extract_features)
    output_audio_features(train_dimAudio)

    test_dimAudio["features"] = test_dimAudio["file_path"].apply(extract_features)
    output_audio_features(test_dimAudio)

#%%
    write_file("3. Model selection")
    model, mean, std = run_CNN(train_dimAudio, test_dimAudio)

#%%
    write_file("4. Model testing")
    evaluate_noise_conditions(model, test_dimAudio, mean, std)

#%%
if __name__ == "__main__":
    main()
# %%
