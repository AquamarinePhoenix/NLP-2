# NLP-2

## Task 2: 10-Word Speech Recognition

### Objective: to develop a simple 10-word speech recognition system and investigate how different noise conditions and signal-to-noise ratios affect its recognition performance.

1. Dataset Preparation

Prepare a speech dataset containing 10 distinct spoken words. You may use an existing speech dataset or create your own dataset by recording the selected words.

Option A: Existing dataset
You may use the Google Speech Commands Dataset or another speech dataset. Select 10 words and create a subset containing recordings from multiple speakers.
The dataset should be reasonably balanced, i.e., the selected words should have approximately the same number of recordings.

Option B: Self-recorded dataset
Create your own dataset using recordings from at least 5 different speakers. Each speaker should pronounce each of the 10 selected words at least 5 times.
Try to keep the recording conditions and audio format consistent across speakers.

### Training and test data
Divide the dataset into training and test sets using a speaker-independent split:

✓ Recordings from the same speaker must not appear in both the training and test sets.
✓ Make sure that all selected words are represented in both the training and test sets.

If you use the Google Speech Commands Dataset, speaker identity can be determined from the recording metadata/file names.

In the report, describe the dataset and its preparation. Include:

✓ the source and name of the dataset, or a brief description of how your own recordings were collected;
✓ the list of the 10 selected words;
✓ the sampling frequency;
✓ the number of speakers;
✓ the total number of recordings and the number of recordings per word;
✓ the procedure used to create the training and test sets;
✓ confirmation that there is no speaker overlap between the training and test sets.

2. Feature Extraction

Convert the audio recordings into a feature representation suitable for speech recognition. Choose one feature representation (e.g., a spectrogram or log-mel spectrogram) and use the same representation and extraction parameters for all recordings.

Before feature extraction, ensure that all audio recordings have the same sampling frequency and consistent signal duration. If necessary, apply appropriate padding.
Specify the parameters used for feature extraction, including: frame/window length, frame shift (hop length), window function, FFT size, number of mel filters (if a mel-based representation is used).

In report:

✓ briefly explain the selected feature representation and justify why it is suitable for speech recognition;
✓ describe the preprocessing and feature extraction procedure;
✓ report all relevant feature extraction parameters;
✓ report the dimensions of the resulting feature representation;
✓ include and interpret a feature plot for at least one selected word.

3. Model selection

Use a single fixed model architecture throughout all experiments. The purpose of the model is to provide a consistent baseline for evaluating the effect of different noise conditions, rather than to optimize the model architecture.

A simple Convolutional Neural Network (CNN) is recommended for spectrogram or log-mel spectrogram input.

Use the speaker-independent training and test split defined during dataset preparation.
In the report:

✓ briefly describe the selected model architecture;
✓ provide the architecture details, including the number and type of layers, activation functions, and output layer;
✓ report the main training parameters, such as loss function, optimizer, number of epochs, and batch size;
✓ describe the training and test sets and confirm that there is no speaker overlap between them.

4. Model testing

Evaluate the trained speech recognition model under different noise conditions. Apply noise only to the test recordings; the training data should remain unchanged.

Evaluate the model under the following conditions:

1. Clean speech (original test recordings without added noise).
2. Synthetic noise (add white Gaussian noise at several SNR levels, for example 30 dB, 20 dB, 10 dB, and 0 dB).
3. Real-world background noise (mix background noise recordings with the test speech at the same SNR levels).

Use the same test recordings for all noise conditions so that the results can be compared directly..
In the report:

✓ include an example waveform and feature representation for clean and noisy speech;
✓ report classification accuracy for clean speech and for each noise type and SNR level;
✓ analyze how recognition performance depends on: SNR, noise type, word identity;
✓ identify which words or word pairs are most frequently confused and discuss how these confusions change under noise;
✓ summarize the main conclusions from the experiment.

### Scope and Limitations:

> [!NOTE]
> The purpose of this task is not to compare or optimize model architectures. A single, simple classification model should be used throughout all experiments.

> The main objective is to investigate the effects of different noise conditions on speech recognition performance: clean speech, synthetic Gaussian noise, and real-world background noise, all at matched signal-to-noise ratios (SNRs).

> The same trained model and test set should be used for all experimental conditions. Recognition accuracy may be modest because model optimization is outside the scope of this task. Conclusions should therefore focus on relative performance differences across noise conditions and SNR levels, rather than on achieving the highest possible recognition accuracy.

## Notes on Task 2

### Dataset

I use the [Google Speech Commands Dataset v0.01](https://www.tensorflow.org/datasets/catalog/speech_commands), which contains short audio recordings of spoken words from multiple speakers.

**Source:** [Google Speech Commands Dataset — Version 1 download](https://storage.googleapis.com/download.tensorflow.org/data/speech_commands_v0.01.tar.gz)

### Power BI-styled plots

`nlplib.plot` provides Matplotlib helpers using the Microsoft Power BI default
palette and visual styling. The source theme is available as
[`nlplib/powerbi_default_theme.json`](nlplib/powerbi_default_theme.json).

```python
from nlplib.plot import histogram_plot, line_chart_plot, scatter_plot

histogram_plot(values, bins=20, title="Audio duration")
scatter_plot(x_values, y_values, xlabel="Duration", ylabel="Energy")
line_chart_plot(
    epochs,
    [train_loss, validation_loss],
    labels=["Train", "Validation"],
)
```

Each function returns a Matplotlib `Axes`, so callers can add annotations,
change limits, or save the figure with the normal Matplotlib API.

For this project, we selected 10 distinct word classes (`zero`–`nine`) and created a balanced subset for speech recognition.

### Experiment protocol

1. **Dataset preparation:** Recordings are collected from the `zero`–`nine`
   directories. The speaker ID is the filename prefix before
   `_nohash_<index>.wav`. `GroupShuffleSplit` creates an 80/20 split by
   speaker, and the split is accepted only when every word occurs in both
   sets and the speaker sets are disjoint.
2. **Feature extraction:** Every recording is loaded as mono audio and
   resampled to 16 kHz, then padded with zeros or truncated to 1 second
   (16,000 samples). A Hann-window log-mel spectrogram is extracted with a
   25 ms frame/window length (400 samples), 10 ms frame shift (160 samples),
   512-point FFT, and 40 mel filters. This produces a `(40, 101)` input.
3. **Model selection:** A single fixed CNN is trained on the clean training
   features for 15 epochs with Adam, batch size 32, and sparse categorical
   cross-entropy. Training-set mean and standard deviation are used to
   normalize all inputs.
4. **Model testing:** The same held-out test recordings are evaluated clean,
   with white Gaussian noise, and with recordings from
   `data/_background_noise_`. Noise is added only during test feature
   generation at 30, 20, 10, and 0 dB SNR. The training features and model
   remain unchanged across all conditions.
