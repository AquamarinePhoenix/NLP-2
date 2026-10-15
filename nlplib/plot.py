from __future__ import annotations
import json
from typing import Any, Iterable, Sequence
import matplotlib.pyplot as plt
from matplotlib.axes import Axes
from matplotlib.patheffects import withSimplePatchShadow
import numpy as np
from nlplib.config import HOP_LENGTH, SAMPLE_RATE, THEME_PATH

def load_powerbi_theme() -> dict[str, Any]:
    with THEME_PATH.open(encoding="utf-8") as theme_file:
        return json.load(theme_file)

def _theme_colors() -> list[str]:
    colors = load_powerbi_theme().get("dataColors")
    if not isinstance(colors, list) or not all(isinstance(color, str) for color in colors):
        raise ValueError(f"Invalid dataColors in {THEME_PATH}")
    return colors

def _prepare_axes(
    ax: Axes | None,
    *,
    title: str | None,
    xlabel: str | None,
    ylabel: str | None,
) -> Axes:
    if ax is None:
        _, ax = plt.subplots()

    ax.set_facecolor("#FFFFFF")
    ax.grid(axis="y", color="#E6E6E6", linewidth=0.8, alpha=0.9)
    ax.set_axisbelow(True)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#B3B3B3")
    ax.spines["bottom"].set_color("#B3B3B3")
    ax.tick_params(colors="#605E5C")
    if title is not None:
        ax.set_title(title, loc="left", color="#252423", pad=14, weight="bold")
    if xlabel is not None:
        ax.set_xlabel(xlabel, color="#605E5C")
    if ylabel is not None:
        ax.set_ylabel(ylabel, color="#605E5C")
    return ax

def _add_shadow(patch: Any) -> None:
    patch.set_path_effects(
        [
            withSimplePatchShadow(
                offset=(2, -2),
                shadow_rgbFace="#000000",
                alpha=0.16,
                rho=0.9,
            )
        ]
    )

def histogram_plot(
    values: Iterable[float],
    *,
    bins: int | Sequence[float] = 10,
    title: str = "Histogram",
    xlabel: str | None = None,
    ylabel: str = "Count",
    color: str | None = None,
    alpha: float = 0.9,
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    axes = _prepare_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    selected_color = color or _theme_colors()[0]
    _, _, patches = axes.hist(
        list(values),
        bins=bins,
        color=selected_color,
        alpha=alpha,
        edgecolor="#FFFFFF",
        linewidth=1,
    )
    for patch in patches:
        _add_shadow(patch)

    if show:
        plt.show()

    return axes

def line_chart_plot(
    x: Iterable[Any],
    y: Iterable[float] | Sequence[Iterable[float]],
    *,
    title: str = "Line chart",
    xlabel: str | None = None,
    ylabel: str | None = None,
    labels: Sequence[str] | None = None,
    colors: Sequence[str] | None = None,
    linewidth: float = 2.5,
    marker: str | None = "o",
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    axes = _prepare_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    x_values = list(x)
    y_values = list(y)
    series: Sequence[Iterable[float]]
    if y_values and isinstance(y_values[0], Iterable) and not isinstance(
        y_values[0], (str, bytes)
    ):
        series = y_values  # type: ignore[assignment]
    else:
        series = [y_values]

    palette = list(colors) if colors is not None else _theme_colors()
    if labels is not None and len(labels) != len(series):
        raise ValueError("labels must contain one label per y series")
    if len(palette) < len(series):
        raise ValueError("colors must contain one color per y series")

    for index, values in enumerate(series):
        axes.plot(
            x_values,
            list(values),
            color=palette[index],
            linewidth=linewidth,
            marker=marker,
            markersize=5 if marker else 0,
            label=labels[index] if labels is not None else None,
        )
    if labels is not None:
        axes.legend(frameon=False)

    if show:
        plt.show()
        
    return axes

def mel_spectrogram_plot(
    spectrogram: np.ndarray | Sequence[Sequence[float]],
    *,
    sample_rate: int = SAMPLE_RATE,
    hop_length: int = HOP_LENGTH,
    title: str = "Mel spectrogram",
    xlabel: str = "Time (s)",
    ylabel: str = "Mel filter",
    cmap: str = "viridis",
    colorbar: bool = True,
    ax: Axes | None = None,
    show: bool = True,
) -> Axes:
    if sample_rate <= 0:
        raise ValueError("sample_rate must be greater than zero")
    if hop_length <= 0:
        raise ValueError("hop_length must be greater than zero")

    values = np.asarray(spectrogram)
    if values.ndim != 2 or values.shape[0] == 0 or values.shape[1] == 0:
        raise ValueError("spectrogram must be a non-empty two-dimensional array")

    duration = values.shape[1] * hop_length / sample_rate
    axes = _prepare_axes(ax, title=title, xlabel=xlabel, ylabel=ylabel)
    axes.grid(False)
    image = axes.imshow(
        values,
        origin="lower",
        aspect="auto",
        extent=(0, duration, 0, values.shape[0]),
        cmap=cmap,
        interpolation="nearest",
    )
    axes.set_xlim(0, duration)
    axes.set_ylim(0, values.shape[0])

    if colorbar:
        bar = axes.figure.colorbar(image, ax=axes, pad=0.02)
        bar.ax.tick_params(colors="#605E5C")
        bar.outline.set_edgecolor("#B3B3B3")
        bar.set_label("Power (dB)", color="#605E5C")

    if show:
        plt.show()

    return axes