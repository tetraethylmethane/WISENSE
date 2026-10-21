"""Figures for paper/wisense_replication.tex.

    python -m tools.make_figures            # writes paper/figures/*.pdf

fig_subcarriers.pdf  accuracy of each of the 52 kept subcarriers on the held-out
                     subjects, authors' released model vs ours -- the evidence
                     for the five "problematic" subcarriers.
fig_training.pdf     training accuracy per epoch across the three resumed
                     Kaggle sessions, with held-out checkpoint scores.

Print figures: one axis each, colour never the only encoding (hatched bands,
direct labels, distinct markers). Palette validated with the dataviz checker.
"""

from __future__ import annotations

import csv
from pathlib import Path

import numpy as np

from wisense.csi import KEPT_SUBCARRIERS, PROBLEM_KEPT
from wisense.data import RAW_DIR, ROOT, load

OUT = ROOT / "paper" / "figures"
RUN = ROOT / "artifacts" / "small_paper"
PUBLISHED = RAW_DIR.parents[1] / "Embedded_Model" / "small_model_complete_EMD_LSTM.keras"

BLUE, ORANGE = "#2a78d6", "#eb6834"        # categorical slots 1-2, validated
INK, INK2, GRID = "#0b0b0b", "#52514e", "#e4e3df"

# Held-out per-subcarrier accuracy of the paper-protocol checkpoints, measured
# with wisense.evaluate after each Kaggle session (epoch 300 is evaluation.json).
CHECKPOINTS = {127: 0.9159, 283: 0.9541, 300: 0.9456}
SESSION_STARTS = (127, 283)


def _style(plt) -> None:
    plt.rcParams.update({
        "font.family": "serif", "font.size": 8, "axes.labelsize": 8,
        "axes.edgecolor": INK2, "axes.labelcolor": INK, "xtick.color": INK2,
        "ytick.color": INK2, "axes.spines.top": False, "axes.spines.right": False,
        "axes.grid": True, "grid.color": GRID, "grid.linewidth": 0.5,
        "legend.frameon": False, "pdf.fonttype": 42,
    })


def per_subcarrier_accuracy(model, x: np.ndarray, y: np.ndarray) -> np.ndarray:
    p = model.predict(x.transpose(0, 2, 1).reshape(-1, x.shape[1], 1),
                      batch_size=512, verbose=0).reshape(len(y), x.shape[2], -1)
    return (p.argmax(-1) == y[:, None]).mean(0)


def fig_subcarriers(plt) -> None:
    from tensorflow import keras

    _, tst = load().split()
    theirs = per_subcarrier_accuracy(keras.models.load_model(PUBLISHED, compile=False), tst.x, tst.y)
    ours = per_subcarrier_accuracy(keras.models.load_model(RUN / "model.keras", compile=False), tst.x, tst.y)

    fig, ax = plt.subplots(figsize=(3.5, 2.1))
    idx = np.arange(len(KEPT_SUBCARRIERS))
    for k in PROBLEM_KEPT:
        ax.axvspan(k - 0.5, k + 0.5, facecolor="#f0efec", edgecolor=INK2, hatch="////",
                   linewidth=0, zorder=0)
    ax.plot(idx, theirs, color=BLUE, lw=1.2, marker="o", ms=2.6, label="Authors' model")
    ax.plot(idx, ours, color=ORANGE, lw=1.2, marker="s", ms=2.6, label="Ours (300 epochs)")
    ticks = [0, 10, 20, 26, 30, 40, 51]
    ax.set_xticks(ticks, [str(KEPT_SUBCARRIERS[t]) for t in ticks])
    ax.set_xlim(-0.8, len(idx) - 0.2)
    ax.set_ylim(0, 1.02)
    ax.set_xlabel("Subcarrier index (raw)")
    ax.set_ylabel("Held-out accuracy")
    ax.annotate("excluded:\nraw 1, 38-41", xy=(27.5, 0.30), xytext=(33, 0.22), fontsize=7,
                color=INK, arrowprops=dict(arrowstyle="-", color=INK2, lw=0.6))
    ax.legend(loc="lower left", fontsize=7, handlelength=1.6, bbox_to_anchor=(0.07, 0.04))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_subcarriers.pdf")
    np.savetxt(OUT / "fig_subcarriers.csv",
               np.column_stack([KEPT_SUBCARRIERS, theirs, ours]), delimiter=",",
               header="raw_subcarrier,released_model,ours", comments="", fmt=["%d", "%.4f", "%.4f"])


def fig_training(plt) -> None:
    rows = [r for r in csv.DictReader(open(RUN / "history.csv")) if r.get("epoch")]
    ep = np.array([int(r["epoch"]) + 1 for r in rows])
    acc = np.array([float(r["accuracy"]) for r in rows])

    fig, ax = plt.subplots(figsize=(3.5, 2.0))
    ax.plot(ep, acc, color=BLUE, lw=1.2, label="Training accuracy")
    ce = sorted(CHECKPOINTS)
    ax.plot(ce, [CHECKPOINTS[e] for e in ce], ls="none", marker="D", ms=4.5, color=ORANGE,
            markeredgecolor="white", markeredgewidth=0.8, label="Held-out, per subcarrier")
    for s in SESSION_STARTS:
        ax.axvline(s + 0.5, color=INK2, lw=0.6, ls="--")
    ax.text(SESSION_STARTS[0] + 3, 0.42, "resume", fontsize=7, color=INK2)
    ax.text(SESSION_STARTS[1] - 3, 0.42, "resume", fontsize=7, color=INK2, ha="right")
    ax.set_xlim(0, 302)
    ax.set_ylim(0.4, 1.0)
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Accuracy")
    ax.legend(loc="center right", fontsize=7, handlelength=1.6, bbox_to_anchor=(1.0, 0.45))
    fig.tight_layout(pad=0.3)
    fig.savefig(OUT / "fig_training.pdf")


def main() -> None:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt

    _style(plt)
    OUT.mkdir(parents=True, exist_ok=True)
    fig_training(plt)
    fig_subcarriers(plt)
    print(f"wrote {OUT}")


if __name__ == "__main__":
    main()
