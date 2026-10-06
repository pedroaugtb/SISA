#!/usr/bin/env python3
"""
Figure S4 - outcome-level distribution of the pooled minority-minus-majority
effects (supplementary material).

Full-width figure. One row per social dimension, one column per measure
(citation count, source URL displacement, answer semantic displacement).
Each dot is one of the 21 primary outcomes, after averaging its
location-specific contrast across Dallas, New York and Los Angeles.

Encoding, in the paper's shared palette:
    orange dot   minority-marked condition larger (effect > 0)
    blue dot     majority-marked condition larger (effect < 0)
    grey dot     exactly zero (|effect| <= 1e-12)
    beneath the dots of every row, a summary line: interquartile range
    (capsule), median (diamond) and mean (open circle).

Dots are placed with a deterministic beeswarm: each dot keeps its exact
x value and is moved vertically just enough not to overlap. Very large ties
(the exact zeros of source URL displacement) are compressed smoothly so the
stack stays inside its row.

Input is the outcome-level table exported by
code/supplement/analyze_outcome_effect_heterogeneity.py; the summary
statistics drawn here are recomputed and checked against its summary table.
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from matplotlib.colors import to_rgb
from matplotlib.lines import Line2D
from matplotlib.patches import FancyBboxPatch, Rectangle


# ============================================================
# LOCAL STYLE
# ============================================================

HERE = Path(__file__).resolve().parent

sys.path.insert(
    0,
    str(HERE),
)

import paper_style as ps  # noqa: E402


# ============================================================
# PATHS
# ============================================================

BASE_DEFAULT = HERE.parents[1]

RESULTS_DEFAULT = (
    BASE_DEFAULT
    / "supplementary_results"
    / "outcome_effect_heterogeneity"
)

OUTPUT_DEFAULT = (
    BASE_DEFAULT
    / "figures_three_locations_pooled"
)

OUTPUT_STEM = "figure_S4_outcome_effect_distributions"


def parse_args():

    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--results-dir",
        type=Path,
        default=RESULTS_DEFAULT,
        help="Folder with outcome_effects_long.csv and the summary table.",
    )

    parser.add_argument(
        "--output-dir",
        type=Path,
        default=OUTPUT_DEFAULT,
    )

    return parser.parse_args()


args = parse_args()


# ============================================================
# LOAD OUTCOME-LEVEL EFFECTS
# ============================================================

effects = pd.read_csv(
    args.results_dir / "outcome_effects_long.csv"
)

reference = pd.read_csv(
    args.results_dir / "outcome_heterogeneity_full_summary.csv"
)


ZERO_TOLERANCE = 1e-12

MEASURES = [

    {
        "id": "citation_count",
        "title": "Citation count",
        "xlim": (-11.5, 13.5),
        "xticks": [-10, -5, 0, 5, 10],
        "decimals": 0,
    },

    {
        "id": "source_url_displacement",
        "title": "Source URL displacement",
        "xlim": (-0.115, 0.475),
        "xticks": [-0.1, 0, 0.1, 0.2, 0.3, 0.4],
        "decimals": 1,
    },

    {
        "id": "answer_semantic_displacement",
        "title": "Answer semantic displacement",
        "xlim": (-0.14, 0.355),
        "xticks": [-0.1, 0, 0.1, 0.2, 0.3],
        "decimals": 1,
    },
]


if len(effects) != len(MEASURES) * len(ps.DIMENSIONS) * 21:
    raise RuntimeError(
        f"Expected {len(MEASURES) * len(ps.DIMENSIONS) * 21} outcome effects, "
        f"found {len(effects)}."
    )


# ============================================================
# SUMMARIES (recomputed and checked against the analysis table)
# ============================================================

def summarise(values):

    values = np.asarray(values, dtype=float)

    q1, median, q3 = np.percentile(values, [25, 50, 75])

    return {
        "mean": values.mean(),
        "median": median,
        "q1": q1,
        "q3": q3,
        "n_positive": int((values > ZERO_TOLERANCE).sum()),
        "n_zero": int((np.abs(values) <= ZERO_TOLERANCE).sum()),
        "n_negative": int((values < -ZERO_TOLERANCE).sum()),
    }


summary = {}

for measure in MEASURES:

    for dimension in ps.DIMENSIONS:

        values = effects.loc[
            effects["measure_id"].eq(measure["id"])
            & effects["dimension"].eq(dimension),
            "outcome_effect",
        ]

        if len(values) != 21:
            raise RuntimeError(f"{measure['id']} / {dimension}: {len(values)} outcomes.")

        stats = summarise(values)

        expected = reference.loc[
            reference["measure_id"].eq(measure["id"])
            & reference["dimension"].eq(dimension)
        ].iloc[0]

        for key in ["mean", "median", "q1", "q3", "n_positive", "n_zero", "n_negative"]:
            if not np.isclose(stats[key], expected[key]):
                raise RuntimeError(
                    f"{measure['id']} / {dimension}: {key} = {stats[key]} "
                    f"differs from the analysis table ({expected[key]})."
                )

        if not (measure["xlim"][0] < values.min() and values.max() < measure["xlim"][1]):
            raise RuntimeError(f"{measure['id']}: x-limits do not cover the data.")

        summary[(measure["id"], dimension)] = stats


# ============================================================
# PRINT VALUES USED IN FIGURE
# ============================================================

print("\nVALUES USED IN FIGURE (21 outcomes per row)\n")

print(
    pd.DataFrame(summary)
    .T
    .rename_axis(["measure", "dimension"])
    .round(4)
    .to_string()
)


# ============================================================
# DRAW
# ============================================================

ps.use_paper_style()


def mix(color, share):
    """`color` blended with white; share = 1 returns the colour."""

    rgb = np.array(to_rgb(color))

    return tuple(share * rgb + (1 - share) * np.ones(3))


SIGN_COLOR = {
    "positive": ps.FOCAL,
    "negative": ps.COMPARISON,
    "zero": ps.GENERIC,
}

IQR_COLOR = mix(ps.INK, 0.55)


# ============================================================
# FIGURE GEOMETRY (inches)
# ============================================================

FIG_W = ps.TEXT_WIDTH

LABEL_W = 1.00
PANEL_GAP = 0.24
RIGHT_MARGIN = 0.06

PANEL_W = (
    FIG_W
    - LABEL_W
    - RIGHT_MARGIN
    - 2 * PANEL_GAP
) / 3

ROW_PITCH = 0.34                 # inches per dimension row
PANEL_BOTTOM = 0.46
PANEL_H = len(ps.DIMENSIONS) * ROW_PITCH
HEADER_H = 0.50

FIG_H = PANEL_BOTTOM + PANEL_H + HEADER_H


# ------------------------------------------------------------
# Inside a row (printed points): the dot cloud sits slightly
# above the row centre, the summary line below it.
# ------------------------------------------------------------

CLOUD_SHIFT_PT = 3.0
CLOUD_CAP_PT = 7.0               # half-height available to the swarm
SUMMARY_SHIFT_PT = -8.0

DOT_PT = 2.5                     # dot diameter
DOT_GAP_PT = 0.35


ROW_Y = np.arange(len(ps.DIMENSIONS))[::-1].astype(float)
YLIM = (-0.5, len(ps.DIMENSIONS) - 0.5)


# ============================================================
# DRAWING PRIMITIVES
# ============================================================

def beeswarm(x_pt, diameter):
    """
    Vertical offsets (points) for dots at horizontal positions `x_pt`:
    each dot, in order of x, takes the offset closest to the row centre
    that does not overlap any dot already placed.
    """

    order = np.argsort(x_pt, kind="stable")
    placed = []
    offsets = np.zeros(len(x_pt))

    for index in order:

        x = x_pt[index]
        near = [(px, py) for px, py in placed if abs(px - x) < diameter]

        candidates = [0.0]

        for px, py in near:
            dy = np.sqrt(diameter ** 2 - (px - x) ** 2)
            candidates.extend([py - dy, py + dy])

        for y in sorted(candidates, key=lambda c: (abs(c), c)):
            if all((x - px) ** 2 + (y - py) ** 2 >= diameter ** 2 - 1e-6 for px, py in near):
                break

        placed.append((x, y))
        offsets[index] = y

    return offsets


def tick_label(value, decimals):

    if np.isclose(value, 0):
        return "0"

    if decimals == 0:
        return f"{value:+.0f}".replace("-", "−")

    return (
        f"{value:+.{decimals}f}"
        .replace("-0.", "−.")
        .replace("+0.", "+.")
    )


def points_to_data_y(ax, pt):
    """Vertical distance in points -> data units of `ax`."""

    height_pt = ax.get_position().height * FIG_H * 72

    return pt / height_pt * (YLIM[1] - YLIM[0])


def capsule(ax, low, high, y, color, width_pt, xlim):
    """Rounded bar whose visible ends sit exactly on `low` and `high`."""

    radius = (width_pt / 2) / (PANEL_W * 72) * (xlim[1] - xlim[0])
    start, end = low + radius, high - radius

    if end < start:
        start = end = (low + high) / 2

    ax.plot(
        [start, end],
        [y, y],
        color=color,
        linewidth=width_pt,
        solid_capstyle="round",
        zorder=3,
    )


# ============================================================
# CREATE FIGURE
# ============================================================

fig = plt.figure(
    figsize=(
        FIG_W,
        FIG_H,
    )
)


axes = [
    ps.add_axes_in(
        fig,
        LABEL_W + i * (PANEL_W + PANEL_GAP),
        PANEL_BOTTOM,
        PANEL_W,
        PANEL_H,
    )
    for i in range(len(MEASURES))
]


# ============================================================
# ROW BANDS AND LABELS (span the whole figure width)
# ============================================================

rows_trans = ps.fig_x_data_y(fig, axes[0])

for index, (dimension, y) in enumerate(zip(ps.DIMENSIONS, ROW_Y)):

    if index % 2 == 0:
        fig.add_artist(
            Rectangle(
                (0, y - 0.5),
                1,
                1,
                transform=rows_trans,
                facecolor=ps.BAND,
                edgecolor="none",
                zorder=-10,
            )
        )

    fig.text(
        ps.fig_x(fig, 0.06),
        y,
        ps.DIMENSION_LABELS[dimension],
        transform=rows_trans,
        ha="left",
        va="center",
        fontsize=ps.FS_LABEL,
        color=ps.INK,
    )


# ============================================================
# PANELS
# ============================================================

for ax, measure in zip(axes, MEASURES):

    xlim = measure["xlim"]

    ax.set_xlim(*xlim)
    ax.set_ylim(*YLIM)
    ax.set_yticks([])

    ax.set_xticks(measure["xticks"])
    ax.xaxis.set_major_formatter(
        plt.FuncFormatter(
            lambda value, _, d=measure["decimals"]: tick_label(value, d)
        )
    )
    ax.tick_params(
        axis="x",
        length=2.5,
        pad=2.0,
        labelsize=ps.FS_NOTE - 0.4,
        labelcolor=ps.MUTED,
    )

    ps.strip_axes(ax, keep=("bottom",))

    for tick in measure["xticks"]:
        if not np.isclose(tick, 0):
            ax.axvline(tick, color=ps.GRID, linewidth=0.45, zorder=0)

    ax.axvline(0, color=ps.INK_SOFT, linewidth=0.65, zorder=1)

    cloud_shift = points_to_data_y(ax, CLOUD_SHIFT_PT)
    summary_shift = points_to_data_y(ax, SUMMARY_SHIFT_PT)
    x_scale_pt = PANEL_W * 72 / (xlim[1] - xlim[0])


    for dimension, y in zip(ps.DIMENSIONS, ROW_Y):

        values = (
            effects.loc[
                effects["measure_id"].eq(measure["id"])
                & effects["dimension"].eq(dimension)
            ]
            .sort_values("outcome_id")["outcome_effect"]
            .to_numpy(float)
        )

        # ----------------------------------------------------
        # Dot cloud: exact x, beeswarm in y, ties compressed
        # ----------------------------------------------------

        offsets_pt = beeswarm(values * x_scale_pt, DOT_PT + DOT_GAP_PT)
        offsets_pt = CLOUD_CAP_PT * np.tanh(offsets_pt / CLOUD_CAP_PT)

        signs = np.where(
            values > ZERO_TOLERANCE,
            "positive",
            np.where(values < -ZERO_TOLERANCE, "negative", "zero"),
        )

        ax.scatter(
            values,
            y + cloud_shift + points_to_data_y(ax, offsets_pt),
            s=DOT_PT ** 2,
            c=[SIGN_COLOR[sign] for sign in signs],
            edgecolors=ps.WHITE,
            linewidths=0.3,
            zorder=2,
        )

        # ----------------------------------------------------
        # Summary line: IQR, mean, median
        # ----------------------------------------------------

        stats = summary[(measure["id"], dimension)]
        line_y = y + summary_shift

        capsule(ax, stats["q1"], stats["q3"], line_y, IQR_COLOR, 2.4, xlim)

        ax.plot(
            [stats["mean"]],
            [line_y],
            marker="o",
            markersize=3.9,
            markerfacecolor=ps.WHITE,
            markeredgecolor=ps.INK,
            markeredgewidth=0.9,
            linestyle="none",
            zorder=4,
        )

        ax.plot(
            [stats["median"]],
            [line_y],
            marker="D",
            markersize=3.9,
            markerfacecolor=ps.INK,
            markeredgecolor=ps.WHITE,
            markeredgewidth=0.6,
            linestyle="none",
            zorder=5,
        )


    # --------------------------------------------------------
    # Column header
    # --------------------------------------------------------

    ax.text(
        0.5,
        1,
        measure["title"],
        transform=ps.offset(ax.transAxes, fig, 0, 7.0),
        ha="center",
        va="baseline",
        fontsize=ps.FS_TITLE - 0.2,
        fontweight="bold",
        color=ps.INK,
    )


# ============================================================
# DIVIDERS: a dark rule in the middle of each gap between
# panels, from the tick labels up to the column titles
# ============================================================

for i in range(1, len(MEASURES)):

    divider_x = LABEL_W + i * PANEL_W + (i - 0.5) * PANEL_GAP

    fig.add_artist(
        Line2D(
            [ps.fig_x(fig, divider_x)] * 2,
            [
                ps.fig_y(fig, PANEL_BOTTOM - 0.17),
                ps.fig_y(fig, PANEL_BOTTOM + PANEL_H + 0.24),
            ],
            transform=fig.transFigure,
            color=ps.INK_SOFT,
            linewidth=0.8,
            zorder=5,
        )
    )


# ============================================================
# BOTTOM: DIRECTION GUIDE
# ============================================================

GUIDE_Y = PANEL_BOTTOM - 0.30

panels_left = LABEL_W
panels_right = LABEL_W + 3 * PANEL_W + 2 * PANEL_GAP

for x, marker, color, text, ha, dx in [
    (panels_left + 0.04, "<", ps.COMPARISON, "majority larger", "left", 0.045),
    (panels_right - 0.04, ">", ps.FOCAL, "minority larger", "right", -0.045),
]:

    fig.add_artist(
        Line2D(
            [ps.fig_x(fig, x)],
            [ps.fig_y(fig, GUIDE_Y)],
            transform=fig.transFigure,
            marker=marker,
            markersize=3.6,
            markerfacecolor=color,
            markeredgecolor=color,
            linestyle="none",
        )
    )

    fig.text(
        ps.fig_x(fig, x + dx),
        ps.fig_y(fig, GUIDE_Y),
        text,
        ha=ha,
        va="center",
        fontsize=ps.FS_NOTE - 0.5,
        color=ps.MUTED,
    )


# ============================================================
# KEY: a boxed 2 x 3 block in the upper-left corner
# ============================================================

KEY_LEFT = 0.04
KEY_W = 1.28
KEY_TOP = FIG_H - 0.06
KEY_H = 0.40

key = ps.add_axes_in(
    fig,
    KEY_LEFT,
    KEY_TOP - KEY_H,
    KEY_W,
    KEY_H,
)

key.set_xlim(0, KEY_W)
key.set_ylim(0, KEY_H)
key.axis("off")

key.add_patch(
    FancyBboxPatch(
        (0.02, 0.02),
        KEY_W - 0.04,
        KEY_H - 0.04,
        boxstyle="round,pad=0,rounding_size=0.035",
        facecolor=ps.WHITE,
        edgecolor=ps.RULE,
        linewidth=0.5,
        zorder=0,
    )
)

ROW_STEP = 0.112
FIRST_ROW = KEY_H - 0.088

for row, (sign, text) in enumerate([
    ("positive", "Minority larger"),
    ("negative", "Majority larger"),
    ("zero", "No difference"),
]):

    y = FIRST_ROW - row * ROW_STEP

    key.scatter(
        0.13,
        y,
        s=(DOT_PT + 0.6) ** 2,
        color=SIGN_COLOR[sign],
        edgecolors=ps.WHITE,
        linewidths=0.3,
        zorder=2,
    )

    key.text(
        0.21,
        y,
        text,
        ha="left",
        va="center",
        fontsize=ps.FS_NOTE - 0.4,
        color=ps.INK_SOFT,
    )


SUMMARY_X = 0.86

key.plot(
    [SUMMARY_X - 0.045, SUMMARY_X + 0.045],
    [FIRST_ROW] * 2,
    color=IQR_COLOR,
    linewidth=2.4,
    solid_capstyle="round",
)

key.plot(
    [SUMMARY_X],
    [FIRST_ROW - ROW_STEP],
    marker="D",
    markersize=3.9,
    markerfacecolor=ps.INK,
    markeredgecolor=ps.WHITE,
    markeredgewidth=0.6,
    linestyle="none",
)

key.plot(
    [SUMMARY_X],
    [FIRST_ROW - 2 * ROW_STEP],
    marker="o",
    markersize=3.9,
    markerfacecolor=ps.WHITE,
    markeredgecolor=ps.INK,
    markeredgewidth=0.9,
    linestyle="none",
)

for row, text in enumerate(["IQR", "Median", "Mean"]):
    key.text(
        SUMMARY_X + 0.09,
        FIRST_ROW - row * ROW_STEP,
        text,
        ha="left",
        va="center",
        fontsize=ps.FS_NOTE - 0.4,
        color=ps.INK_SOFT,
    )


# ============================================================
# SAVE
# ============================================================

ps.save_to(
    fig,
    args.output_dir,
    OUTPUT_STEM,
)
