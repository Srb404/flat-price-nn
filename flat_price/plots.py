"""Figures for the README, rendered in a light and a dark variant for GitHub themes."""

import json
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402

THEMES = {
    "light": {
        "surface": "#fcfcfb",
        "text": "#0b0b0b",
        "text2": "#52514e",
        "grid": "#e4e3df",
        "series": ["#2a78d6", "#eb6834"],
    },
    "dark": {
        "surface": "#1a1a19",
        "text": "#ffffff",
        "text2": "#c3c2b7",
        "grid": "#383835",
        "series": ["#3987e5", "#d95926"],
    },
}


CITY_NAMES = {
    "bialystok": "Białystok",
    "czestochowa": "Częstochowa",
    "gdansk": "Gdańsk",
    "krakow": "Kraków",
    "lodz": "Łódź",
    "poznan": "Poznań",
    "rzeszow": "Rzeszów",
    "wroclaw": "Wrocław",
}


def _style(ax, t):
    ax.set_facecolor(t["surface"])
    ax.figure.set_facecolor(t["surface"])
    for side in ("top", "right"):
        ax.spines[side].set_visible(False)
    for side in ("left", "bottom"):
        ax.spines[side].set_color(t["grid"])
    ax.tick_params(colors=t["text2"], labelsize=9)
    ax.grid(color=t["grid"], linewidth=0.8)
    ax.set_axisbelow(True)
    ax.xaxis.label.set_color(t["text2"])
    ax.yaxis.label.set_color(t["text2"])


def learning_curve(history, t):
    fig, ax = plt.subplots(figsize=(7, 4))
    _style(ax, t)
    epochs = np.arange(1, len(history["loss"]) + 1)
    best = int(np.argmin(history["val_loss"]))
    for key, label, color in [
        ("loss", "train", t["series"][0]),
        ("val_loss", "validation", t["series"][1]),
    ]:
        ax.plot(epochs, history[key], color=color, linewidth=2, label=label)
        ax.annotate(
            label,
            (epochs[-1], history[key][-1]),
            xytext=(6, 0),
            textcoords="offset points",
            color=t["text2"],
            fontsize=9,
            va="center",
        )
    ax.axvline(best + 1, color=t["text2"], linewidth=1, linestyle=":")
    ax.annotate(
        f"best epoch {best + 1}",
        (best + 1, ax.get_ylim()[1]),
        xytext=(4, -12),
        textcoords="offset points",
        color=t["text2"],
        fontsize=9,
    )
    ax.set_xlabel("epoch")
    ax.set_ylabel("MSE on standardized log(price)")
    ax.set_title("Learning curve", loc="left", fontsize=12, color=t["text"])
    legend = ax.legend(frameon=False, labelcolor=t["text2"], fontsize=9, loc="upper right")
    legend.set_bbox_to_anchor((1, 0.9))
    return fig


def predicted_vs_actual(pred, t):
    fig, ax = plt.subplots(figsize=(6, 6))
    _style(ax, t)
    actual, predicted = pred["price"] / 1e6, pred["predicted"] / 1e6
    ax.scatter(actual, predicted, s=8, color=t["series"][0], alpha=0.25, linewidths=0)
    lim = [0.15, 3.3]
    ax.plot(lim, lim, color=t["text2"], linewidth=1, linestyle="--")
    ax.annotate(
        "perfect prediction",
        (2.2, 2.2),
        xytext=(6, -14),
        textcoords="offset points",
        color=t["text2"],
        fontsize=9,
        rotation=45,
    )
    ax.set_xscale("log")
    ax.set_yscale("log")
    ticks = [0.2, 0.5, 1, 2, 3]
    ax.set_xticks(ticks, [f"{v:g}" for v in ticks])
    ax.set_yticks(ticks, [f"{v:g}" for v in ticks])
    ax.minorticks_off()
    ax.set_xlim(lim)
    ax.set_ylim(lim)
    ax.set_xlabel("actual price [mln PLN]")
    ax.set_ylabel("predicted price [mln PLN]")
    ax.set_title("Neural network on the test set", loc="left", fontsize=12, color=t["text"])
    return fig


def error_by_city(pred, t):
    ape = pred.assign(
        nn=(pred["predicted"] - pred["price"]).abs() / pred["price"] * 100,
        lin=(pred["linear"] - pred["price"]).abs() / pred["price"] * 100,
    )
    by_city = ape.groupby("city")[["nn", "lin"]].median().sort_values("nn", ascending=False)
    fig, ax = plt.subplots(figsize=(7, 6))
    _style(ax, t)
    ax.grid(axis="y", visible=False)
    y = np.arange(len(by_city))
    h = 0.38
    for offset, col, label, color in [
        (h / 2 + 0.01, "nn", "neural network", t["series"][0]),
        (-h / 2 - 0.01, "lin", "linear regression", t["series"][1]),
    ]:
        ax.barh(y + offset, by_city[col], height=h, color=color, label=label)
    for yi, v in zip(y, by_city["nn"], strict=True):
        ax.annotate(
            f"{v:.1f}%",
            (v, yi + h / 2),
            xytext=(4, 0),
            textcoords="offset points",
            va="center",
            color=t["text2"],
            fontsize=8,
        )
    ax.set_yticks(y, [CITY_NAMES.get(c, c.capitalize()) for c in by_city.index])
    ax.set_xlabel("median absolute percentage error [%]")
    ax.set_title("Error by city (test set)", loc="left", fontsize=12, color=t["text"])
    ax.legend(frameon=False, labelcolor=t["text2"], fontsize=9, loc="lower right")
    return fig


def make_figures(reports: Path) -> None:
    out = reports / "figures"
    out.mkdir(parents=True, exist_ok=True)
    history = json.loads((reports / "history.json").read_text())
    pred = pd.read_csv(reports / "test_predictions.csv")
    for mode, t in THEMES.items():
        for name, fig in [
            ("learning_curve", learning_curve(history, t)),
            ("predicted_vs_actual", predicted_vs_actual(pred, t)),
            ("error_by_city", error_by_city(pred, t)),
        ]:
            fig.tight_layout()
            fig.savefig(out / f"{name}_{mode}.png", dpi=150, facecolor=t["surface"])
            plt.close(fig)


if __name__ == "__main__":
    make_figures(Path("reports"))
