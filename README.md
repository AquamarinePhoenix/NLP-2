# NLP-2

Lab2 & Lab3

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
