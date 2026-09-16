"""Figures for the report.

Design decisions, recorded so the figures stay consistent as the experiments grow.

* **One light palette, committed deliberately.** These are print figures embedded
  in a Word report, so there is no viewer theme to respond to. The three
  categorical hues are the first three slots of a validated palette -- they clear
  the all-pairs colour-vision-deficiency and normal-vision separation floors, so
  the three classes stay distinguishable in greyscale printing and for a
  colourblind reader.
* **Never two y-axes.** Loss and accuracy live on different scales, so the
  training curves are two stacked panels rather than one chart with a twin axis.
  A twin axis lets the reader infer a crossover that is an artefact of the
  arbitrary relative scaling.
* **Direct labels, not colour alone.** Every series is labelled at its end or on
  its bar. One of the three hues sits below 3:1 contrast against the page, and
  identity must never depend on a colour a reader may not resolve.
* **Every figure writes a CSV beside it.** The numbers in the report tables and
  the numbers in the figures then cannot drift apart, and the data stays readable
  by anyone who cannot see the image.
"""

from __future__ import annotations

import csv
from pathlib import Path
from typing import Any

import matplotlib

matplotlib.use("Agg")  # no display in CI or over SSH
import matplotlib.pyplot as plt  # noqa: E402
from matplotlib.colors import LinearSegmentedColormap  # noqa: E402
from matplotlib.patches import BoxStyle, FancyBboxPatch  # noqa: E402

from antod.data.synth import LABEL_NAMES  # noqa: E402

# --------------------------------------------------------------------------- #
# palette
# --------------------------------------------------------------------------- #
SURFACE = "#fcfcfb"
INK = "#0b0b0b"
INK_SOFT = "#52514e"
GRID = "#e4e3df"

#: First three slots of the validated categorical order: blue, orange, aqua.
SERIES = ["#2a78d6", "#eb6834", "#1baf7a", "#eda100", "#e87ba4", "#4a3aa7"]

#: Single-hue blue ramp for magnitude (confusion matrices, importances).
SEQUENTIAL = ["#cde2fb", "#9ec5f4", "#6da7ec", "#3987e5", "#256abf", "#184f95", "#0d366b"]
BLUES = LinearSegmentedColormap.from_list("antod_blues", SEQUENTIAL)

DPI = 200


def apply_style() -> None:
    """Recessive axes, thin marks, generous whitespace."""
    plt.rcParams.update(
        {
            "figure.facecolor": SURFACE,
            "axes.facecolor": SURFACE,
            "savefig.facecolor": SURFACE,
            "axes.edgecolor": GRID,
            "axes.linewidth": 0.8,
            "axes.labelcolor": INK_SOFT,
            "axes.titlecolor": INK,
            "axes.titleweight": "bold",
            "axes.titlesize": 11,
            "axes.labelsize": 9.5,
            "axes.grid": True,
            "axes.axisbelow": True,
            "grid.color": GRID,
            "grid.linewidth": 0.7,
            "xtick.color": INK_SOFT,
            "ytick.color": INK_SOFT,
            "xtick.labelsize": 8.5,
            "ytick.labelsize": 8.5,
            "xtick.direction": "out",
            "ytick.direction": "out",
            "legend.frameon": False,
            "legend.fontsize": 8.5,
            "lines.linewidth": 2.0,
            "lines.markersize": 5.0,
            "lines.solid_capstyle": "round",
            "font.size": 9.5,
            "figure.dpi": DPI,
            "savefig.dpi": DPI,
            "savefig.bbox": "tight",
        }
    )


def _finish(fig: plt.Figure, ax: plt.Axes | list[plt.Axes]) -> None:
    for a in [ax] if isinstance(ax, plt.Axes) else ax:
        a.spines["top"].set_visible(False)
        a.spines["right"].set_visible(False)
    fig.tight_layout()


def _save(fig: plt.Figure, path: str | Path) -> Path:
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(path)
    plt.close(fig)
    return path


def _write_csv(path: str | Path, header: list[str], rows: list[list[Any]]) -> Path:
    """Table view of a figure, written beside the image."""
    path = Path(path).with_suffix(".csv")
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", newline="", encoding="utf-8") as handle:
        writer = csv.writer(handle)
        writer.writerow(header)
        writer.writerows(rows)
    return path


def _rounded_barh(
    ax: plt.Axes,
    y: float,
    width: float,
    height: float,
    color: str,
    rounding: float,
) -> None:
    """Horizontal bar with a rounded data end and a square baseline end.

    The patch starts one rounding-radius *left* of zero so that its left corners
    fall outside the axes and are clipped away, which leaves the bar flush against
    the baseline while its data end stays rounded.
    """
    ax.add_patch(
        FancyBboxPatch(
            (-rounding, y - height / 2),
            max(width, 1e-9) + rounding,
            height,
            boxstyle=BoxStyle("Round", pad=0, rounding_size=rounding),
            linewidth=0,
            facecolor=color,
            clip_on=True,
            mutation_aspect=1,
        )
    )


# --------------------------------------------------------------------------- #
# confusion matrix
# --------------------------------------------------------------------------- #
def plot_confusion_matrix(
    confusion: list[list[int]],
    path: str | Path,
    title: str = "Confusion matrix",
    labels: list[str] | None = None,
    normalise: bool = True,
) -> Path:
    """Row-normalised confusion heatmap with counts annotated.

    Rows are normalised, so each row reads as "of the flows that really were X,
    where did they go?" -- which is the recall-oriented question. Raw counts stay
    printed in the cells because a normalised matrix alone hides how many flows
    each row rests on.
    """
    apply_style()
    labels = labels or LABEL_NAMES
    counts = [[int(v) for v in row] for row in confusion]
    totals = [max(sum(row), 1) for row in counts]
    shown = (
        [[v / t for v in row] for row, t in zip(counts, totals, strict=True)]
        if normalise
        else counts
    )

    fig, ax = plt.subplots(figsize=(5.4, 4.4))
    image = ax.imshow(shown, cmap=BLUES, vmin=0, vmax=1 if normalise else None)
    ax.grid(False)

    ax.set_xticks(range(len(labels)), [n.replace("_", "\n") for n in labels])
    ax.set_yticks(range(len(labels)), [n.replace("_", "\n") for n in labels])
    ax.set_xlabel("predicted")
    ax.set_ylabel("actual")
    ax.set_title(title)

    for i, row in enumerate(counts):
        for j, count in enumerate(row):
            fraction = count / totals[i]
            # keep cell text legible whichever end of the ramp the cell sits on
            colour = "#ffffff" if (fraction if normalise else 0.0) > 0.55 else INK
            ax.text(
                j,
                i,
                f"{fraction:.1%}\n{count}" if normalise else f"{count}",
                ha="center",
                va="center",
                color=colour,
                fontsize=8.5,
            )

    bar = fig.colorbar(image, ax=ax, fraction=0.046, pad=0.04)
    bar.outline.set_visible(False)
    bar.set_label("share of actual class" if normalise else "flows", color=INK_SOFT, fontsize=8.5)

    _finish(fig, ax)
    _write_csv(
        path,
        ["actual", *labels],
        [[labels[i], *row] for i, row in enumerate(counts)],
    )
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# training curves
# --------------------------------------------------------------------------- #
def plot_training_curves(
    history: list[dict[str, float]],
    path: str | Path,
    title: str = "Training history",
) -> Path:
    """Loss on one panel, the bounded metrics on another.

    Two panels rather than a twin axis: loss and accuracy have unrelated scales,
    and overlaying them invites the reader to see a crossover that is purely an
    artefact of how the two axes were scaled against each other.
    """
    apply_style()
    epochs = [h["epoch"] for h in history]

    fig, axes = plt.subplots(2, 1, figsize=(6.4, 5.6), sharex=True)
    top, bottom = axes

    top.plot(epochs, [h["train_loss"] for h in history], color=SERIES[0])
    top.set_ylabel("training loss")
    top.set_title(title)
    top.annotate(
        "train loss",
        (epochs[-1], history[-1]["train_loss"]),
        textcoords="offset points",
        xytext=(6, 0),
        color=SERIES[0],
        fontsize=8.5,
        va="center",
    )

    tracked = [
        ("val_acc", "val accuracy", SERIES[0]),
        ("val_macro_f1", "val macro-F1", SERIES[1]),
        ("val_obf_recall", "obfuscated recall", SERIES[2]),
    ]
    for key, label, colour in tracked:
        if key not in history[0]:
            continue
        values = [h[key] for h in history]
        bottom.plot(epochs, values, color=colour, label=label)
        bottom.annotate(
            label,
            (epochs[-1], values[-1]),
            textcoords="offset points",
            xytext=(6, 0),
            color=colour,
            fontsize=8.5,
            va="center",
        )

    bottom.set_xlabel("epoch")
    bottom.set_ylabel("score")
    bottom.set_ylim(0, 1.02)
    bottom.legend(loc="lower right", ncols=1)

    _finish(fig, list(axes))
    fig.subplots_adjust(right=0.80)

    columns = [k for k in history[0]]
    _write_csv(path, columns, [[h[c] for c in columns] for h in history])
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# robustness
# --------------------------------------------------------------------------- #
def plot_robustness_curves(
    curves: dict[str, tuple[list[float], list[float]]],
    path: str | Path,
    title: str = "Accuracy under attack",
    ylabel: str = "accuracy",
    xlabel: str = "perturbation budget  $\\epsilon$",
) -> Path:
    """One line per model or defense: metric against attack strength."""
    apply_style()
    fig, ax = plt.subplots(figsize=(6.4, 4.2))

    for i, (label, (xs, ys)) in enumerate(curves.items()):
        colour = SERIES[i % len(SERIES)]
        ax.plot(xs, ys, color=colour, marker="o", label=label)
        ax.annotate(
            label,
            (xs[-1], ys[-1]),
            textcoords="offset points",
            xytext=(7, 0),
            color=colour,
            fontsize=8.5,
            va="center",
        )

    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_ylim(0, 1.02)
    ax.set_title(title)
    if len(curves) > 1:
        ax.legend(loc="lower left")

    _finish(fig, ax)
    fig.subplots_adjust(right=0.74)

    rows: list[list[Any]] = []
    for label, (xs, ys) in curves.items():
        rows += [[label, x, y] for x, y in zip(xs, ys, strict=True)]
    _write_csv(path, ["series", "epsilon", ylabel], rows)
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# ranked bars
# --------------------------------------------------------------------------- #
def plot_ranked_bars(
    values: dict[str, float],
    path: str | Path,
    title: str,
    xlabel: str = "recall",
    top_k: int | None = None,
    ascending: bool = False,
    highlight_below: float | None = None,
) -> Path:
    """Horizontal ranked bars with values printed on each bar.

    Horizontal because the category names (``protocol_mimicry``,
    ``constant_rate_shaping``) do not fit under a vertical axis without rotating
    the labels. Sorted because the ranking *is* the finding -- the reader wants to
    know which technique is worst, not to look them up alphabetically.

    ``highlight_below`` paints bars under a threshold in the second categorical
    hue, so "these are the techniques that defeat the detector" survives greyscale
    printing via the printed value beside each bar.
    """
    apply_style()
    items = sorted(values.items(), key=lambda kv: kv[1], reverse=not ascending)
    if top_k is not None:
        items = items[:top_k]
    items = list(reversed(items))  # matplotlib draws the y axis bottom-up

    height = max(2.4, 0.34 * len(items) + 1.2)
    fig, ax = plt.subplots(figsize=(6.6, height))

    names = [name for name, _ in items]
    scores = [score for _, score in items]
    span = max(scores) if scores else 1.0
    rounding = min(0.012 * span if span else 0.01, 0.16)

    for i, (_name, score) in enumerate(items):
        colour = SERIES[1] if highlight_below is not None and score < highlight_below else SERIES[0]
        # 2px surface gap between neighbouring bars: height 0.66 of the slot
        _rounded_barh(ax, i, score, 0.66, colour, rounding)
        ax.text(
            score + span * 0.015,
            i,
            f"{score:.3f}",
            va="center",
            ha="left",
            fontsize=8.5,
            color=INK_SOFT,
        )

    ax.set_yticks(range(len(items)), [n.replace("_", " ") for n in names])
    ax.set_xlim(0, span * 1.16)
    ax.set_ylim(-0.6, len(items) - 0.4)
    ax.set_xlabel(xlabel)
    ax.set_title(title)
    ax.grid(axis="y", visible=False)

    if highlight_below is not None:
        ax.axvline(highlight_below, color=INK_SOFT, linewidth=1.0, linestyle=(0, (4, 3)))
        ax.annotate(
            f"threshold {highlight_below:g}",
            (highlight_below, len(items) - 0.5),
            textcoords="offset points",
            xytext=(4, -4),
            fontsize=8,
            color=INK_SOFT,
        )

    _finish(fig, ax)
    _write_csv(path, ["name", xlabel], [[n, s] for n, s in reversed(items)])
    return _save(fig, path)


# --------------------------------------------------------------------------- #
# model comparison
# --------------------------------------------------------------------------- #
def plot_model_comparison(
    rows: list[dict[str, Any]],
    path: str | Path,
    metrics: tuple[str, ...] = ("accuracy", "macro_f1", "obfuscated_recall"),
    name_key: str = "model",
    title: str = "Model comparison",
) -> Path:
    """Grouped bars: one group per model, one bar per metric."""
    apply_style()
    models = [str(r[name_key]) for r in rows]
    n_metrics = len(metrics)
    slot = 0.8 / n_metrics

    fig, ax = plt.subplots(figsize=(max(6.4, 1.5 * len(models) + 2.0), 4.2))

    for m, metric in enumerate(metrics):
        offsets = [i - 0.4 + slot * (m + 0.5) for i in range(len(models))]
        heights = [float(r.get(metric, 0.0)) for r in rows]
        ax.bar(
            offsets,
            heights,
            width=slot * 0.86,  # the 14% gap is the 2px surface separation
            color=SERIES[m % len(SERIES)],
            label=metric.replace("_", " "),
        )
        for x, h in zip(offsets, heights, strict=True):
            ax.text(x, h + 0.012, f"{h:.3f}", ha="center", fontsize=7.5, color=INK_SOFT)

    ax.set_xticks(range(len(models)), models, rotation=0)
    ax.set_ylim(0, 1.08)
    ax.set_ylabel("score")
    ax.set_title(title)
    ax.grid(axis="x", visible=False)
    ax.legend(loc="lower right", ncols=n_metrics)

    _finish(fig, ax)
    _write_csv(
        path,
        [name_key, *metrics],
        [[r[name_key], *[r.get(m, "") for m in metrics]] for r in rows],
    )
    return _save(fig, path)
