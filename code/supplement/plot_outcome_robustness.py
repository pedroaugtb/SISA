#!/usr/bin/env python3
"""Paper figure and supplementary tables for outcome-selection robustness."""

from __future__ import annotations

import argparse
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/shared"))

from figure_common import (  # noqa: E402
    DIMENSION_ORDER,
    COL_BG,
    COL_COMPARISON,
    COL_FOCAL,
    COL_GRID,
    COL_INK,
    COL_MUTED,
    COL_PATHWAY,
    COL_SYNTHESIS,
    padded_xlim,
    save_figure,
)


PRIMARY_COLOR = COL_COMPARISON
ROBUSTNESS_COLOR = COL_FOCAL

ANALYSES = [
    "citation_count",
    "source_url_jaccard_distance",
    "answer_semantic_displacement",
    "evidence_semantic_gap",
]

PANEL_LABELS = {
    "citation_count": "Citation count",
    "source_url_jaccard_distance": "Source URL displacement",
    "answer_semantic_displacement": "Answer semantic displacement",
    "evidence_semantic_gap": "Evidence semantic gap",
}

# Wrapped panel headings keep every title inside its own axes width when the
# figure is reduced to a two-column paper layout.
PANEL_TITLES = {
    "citation_count": "Citation count\n\u00a0",
    "source_url_jaccard_distance": "Source URL\ndisplacement",
    "answer_semantic_displacement": "Answer semantic\ndisplacement",
    "evidence_semantic_gap": "Evidence semantic\ngap",
}

ANATOMY_PANEL_SPECS = {
    "citation_count": {
        "title": "CITATION COUNT\n\u00a0",
        "color": COL_FOCAL,
        "left": "Majority\nmore",
        "right": "Minority\nmore",
    },
    "source_url_jaccard_distance": {
        "title": "SOURCE URL\nDISPLACEMENT",
        "color": COL_PATHWAY,
        "left": "Majority\nfarther",
        "right": "Minority\nfarther",
    },
    "answer_semantic_displacement": {
        "title": "ANSWER SEMANTIC\nDISPLACEMENT",
        "color": COL_SYNTHESIS,
        "left": "Majority\nfarther",
        "right": "Minority\nfarther",
    },
}

DISPLAY_DIMENSIONS = {
    "Race": "Race",
    "Ethnicity": "Ethnicity",
    "Gender": "Gender",
    "Disability": "Disability",
    "Sexual Orientation": "Sexual orientation",
    "Gender Identity": "Gender identity",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "robustness_results/paper_outputs",
    )
    return parser.parse_args()


def load_pooled(base: Path) -> pd.DataFrame:
    paths = {
        "Primary outcomes (n=21)": base
        / "results/three_location_pooled_secondary/pooled_dimension_tests.csv",
        "Robustness outcomes (n=7)": base
        / "robustness_results/three_location_pooled_secondary/pooled_dimension_tests.csv",
    }
    frames = []
    for outcome_set, path in paths.items():
        data = pd.read_csv(path)
        data["outcome_set"] = outcome_set
        frames.append(data)
    pooled = pd.concat(frames, ignore_index=True)
    pooled = pooled.loc[
        pooled["analysis"].isin(ANALYSES) & pooled["dimension"].isin(DIMENSION_ORDER)
    ].copy()
    expected = len(ANALYSES) * len(DIMENSION_ORDER) * 2
    if len(pooled) != expected:
        raise RuntimeError(f"Expected {expected} pooled rows; found {len(pooled)}")
    return pooled


def load_location_comparison(base: Path) -> pd.DataFrame:
    relative = Path("three_location_summary/primary_dimension_effects_all_locations.csv")
    primary = pd.read_csv(base / "results" / relative)
    robust = pd.read_csv(base / "robustness_results" / relative)
    keys = ["analysis", "location", "dimension"]
    merged = primary.merge(
        robust,
        on=keys,
        how="outer",
        suffixes=("_primary", "_robustness"),
        validate="one_to_one",
        indicator=True,
    )
    if not merged["_merge"].eq("both").all():
        raise RuntimeError("Primary and robustness location tables do not align")
    merged = merged.drop(columns="_merge")
    primary_effect = "mean_difference_focal_minus_comparison_primary"
    robust_effect = "mean_difference_focal_minus_comparison_robustness"
    merged["direction_consistent"] = (
        np.sign(merged[primary_effect]) == np.sign(merged[robust_effect])
    )
    merged["confidence_intervals_overlap"] = (
        np.maximum(merged["ci_low_primary"], merged["ci_low_robustness"])
        <= np.minimum(merged["ci_high_primary"], merged["ci_high_robustness"])
    )
    return merged


def merge_pooled_sets(pooled: pd.DataFrame) -> pd.DataFrame:
    primary = pooled.loc[pooled["outcome_set"].str.startswith("Primary")].drop(
        columns="outcome_set"
    )
    robust = pooled.loc[pooled["outcome_set"].str.startswith("Robustness")].drop(
        columns="outcome_set"
    )
    keys = ["analysis", "dimension"]
    merged = primary.merge(
        robust,
        on=keys,
        suffixes=("_primary", "_robustness"),
        validate="one_to_one",
    )
    merged["direction_consistent"] = (
        np.sign(merged["mean_difference_primary"])
        == np.sign(merged["mean_difference_robustness"])
    )
    merged["confidence_intervals_overlap"] = (
        np.maximum(merged["bootstrap_ci_low_primary"], merged["bootstrap_ci_low_robustness"])
        <= np.minimum(merged["bootstrap_ci_high_primary"], merged["bootstrap_ci_high_robustness"])
    )
    return merged


def draw_panel(
    ax,
    pooled: pd.DataFrame,
    analysis: str,
    *,
    show_ylabels: bool,
) -> None:
    subset = pooled.loc[pooled["analysis"].eq(analysis)].copy()
    positions = {
        dimension: len(DIMENSION_ORDER) - 1 - index
        for index, dimension in enumerate(DIMENSION_ORDER)
    }
    styles = {
        "Primary outcomes (n=21)": {
            "color": PRIMARY_COLOR,
            "marker": "o",
            "offset": +0.115,
        },
        "Robustness outcomes (n=7)": {
            "color": ROBUSTNESS_COLOR,
            "marker": "D",
            "offset": -0.115,
        },
    }
    values = (
        subset["bootstrap_ci_low"].tolist()
        + subset["bootstrap_ci_high"].tolist()
        + subset["mean_difference"].tolist()
    )
    min_pad = 0.20 if analysis == "citation_count" else 0.008
    xlim = padded_xlim(values, zero=True, min_pad=min_pad)
    ax.axvline(0, color=COL_MUTED, linewidth=0.9, linestyle="--", alpha=0.9, zorder=1)

    for outcome_set, style in styles.items():
        data = subset.loc[subset["outcome_set"].eq(outcome_set)].set_index("dimension")
        for dimension in DIMENSION_ORDER:
            row = data.loc[dimension]
            y = positions[dimension] + style["offset"]
            effect = float(row["mean_difference"])
            low = float(row["bootstrap_ci_low"])
            high = float(row["bootstrap_ci_high"])
            ax.hlines(y, low, high, color=style["color"], linewidth=1.45, zorder=2)
            ax.scatter(
                effect,
                y,
                s=31,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.45,
                zorder=3,
            )

    yticks = [positions[dimension] for dimension in DIMENSION_ORDER]
    ax.set_yticks(yticks)
    if show_ylabels:
        ax.set_yticklabels(
            [DISPLAY_DIMENSIONS[value] for value in DIMENSION_ORDER],
            fontsize=8.2,
            color=COL_INK,
        )
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.7, colors=COL_MUTED)
    ax.set_ylim(-0.55, len(DIMENSION_ORDER) - 0.45)
    ax.set_xlim(*xlim)
    ax.set_title(
        PANEL_TITLES[analysis],
        loc="left",
        fontsize=10.2,
        fontweight="bold",
        color=COL_INK,
        pad=7,
        linespacing=0.95,
        multialignment="left",
    )
    ax.grid(False)
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color(COL_GRID)
    ax.spines["bottom"].set_color(COL_GRID)
    ax.spines["left"].set_linewidth(0.7)
    ax.spines["bottom"].set_linewidth(0.7)


def legend_handles() -> list[Line2D]:
    return [
        Line2D(
            [0], [0], marker="o", color=PRIMARY_COLOR, linewidth=1.45,
            markerfacecolor=PRIMARY_COLOR, markeredgecolor="white", markersize=6.0,
            label="Primary outcomes (n=21)",
        ),
        Line2D(
            [0], [0], marker="D", color=ROBUSTNESS_COLOR, linewidth=1.45,
            markerfacecolor=ROBUSTNESS_COLOR, markeredgecolor="white", markersize=5.5,
            label="Robustness outcomes (n=7)",
        ),
    ]


def draw_anatomy_panel(
    ax,
    pooled: pd.DataFrame,
    analysis: str,
    *,
    show_ylabels: bool,
) -> None:
    """Draw an alternative panel following the visual grammar of Figure 4.3."""
    subset = pooled.loc[pooled["analysis"].eq(analysis)].copy()
    positions = {
        dimension: len(DIMENSION_ORDER) - 1 - index
        for index, dimension in enumerate(DIMENSION_ORDER)
    }
    styles = {
        "Primary outcomes (n=21)": {
            "color": PRIMARY_COLOR,
            "marker": "o",
            "offset": +0.115,
        },
        "Robustness outcomes (n=7)": {
            "color": ROBUSTNESS_COLOR,
            "marker": "D",
            "offset": -0.115,
        },
    }
    values = (
        subset["bootstrap_ci_low"].tolist()
        + subset["bootstrap_ci_high"].tolist()
        + subset["mean_difference"].tolist()
    )
    min_pad = 0.20 if analysis == "citation_count" else 0.008
    xlim = padded_xlim(values, zero=True, min_pad=min_pad)

    # The same low-contrast directional background used by the paper figures.
    ax.axvspan(xlim[0], 0, color=COL_FOCAL, alpha=0.025, zorder=0)
    ax.axvspan(0, xlim[1], color=COL_COMPARISON, alpha=0.018, zorder=0)
    ax.axvline(0, color="#000000", linewidth=0.65, zorder=1)

    for outcome_set, style in styles.items():
        data = subset.loc[subset["outcome_set"].eq(outcome_set)].set_index("dimension")
        for dimension in DIMENSION_ORDER:
            row = data.loc[dimension]
            y = positions[dimension] + style["offset"]
            effect = float(row["mean_difference"])
            low = float(row["bootstrap_ci_low"])
            high = float(row["bootstrap_ci_high"])
            ax.hlines(
                y,
                low,
                high,
                color=style["color"],
                linewidth=1.55,
                zorder=2,
            )
            ax.scatter(
                effect,
                y,
                s=35,
                marker=style["marker"],
                facecolor=style["color"],
                edgecolor="white",
                linewidth=0.5,
                zorder=3,
            )

    yticks = [positions[dimension] for dimension in DIMENSION_ORDER]
    ax.set_yticks(yticks)
    if show_ylabels:
        ax.set_yticklabels(
            [DISPLAY_DIMENSIONS[value] for value in DIMENSION_ORDER],
            fontsize=8.3,
            color=COL_INK,
        )
    else:
        ax.set_yticklabels([])
    ax.tick_params(axis="y", length=0)
    ax.tick_params(axis="x", labelsize=7.8, colors=COL_MUTED)
    ax.set_ylim(-0.55, len(DIMENSION_ORDER) - 0.45)
    ax.set_xlim(*xlim)

    spec = ANATOMY_PANEL_SPECS[analysis]
    ax.set_title(
        spec["title"],
        loc="left",
        fontsize=9.5,
        fontweight="bold",
        color=COL_INK,
        pad=24,
        linespacing=0.94,
        multialignment="left",
    )
    ax.text(
        0.01,
        1.014,
        spec["left"],
        transform=ax.transAxes,
        ha="left",
        va="bottom",
        fontsize=8.0,
        fontweight="normal",
        color=COL_INK,
        linespacing=0.90,
        multialignment="left",
    )
    ax.text(
        0.99,
        1.014,
        spec["right"],
        transform=ax.transAxes,
        ha="right",
        va="bottom",
        fontsize=8.0,
        fontweight="normal",
        color=COL_INK,
        linespacing=0.90,
        multialignment="right",
    )
    for spine in ax.spines.values():
        spine.set_visible(False)
    ax.grid(axis="x", color="white", linewidth=0.55, alpha=1.0, zorder=0)


def plot_anatomy_style_figure(pooled: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.50, 3.85), facecolor=COL_BG)
    for index, (ax, analysis) in enumerate(zip(axes, ANALYSES[:3])):
        draw_anatomy_panel(ax, pooled, analysis, show_ylabels=index == 0)
    fig.legend(
        handles=legend_handles(),
        loc="upper center",
        bbox_to_anchor=(0.53, 0.992),
        ncol=2,
        frameon=False,
        fontsize=8.2,
        handletextpad=0.55,
        columnspacing=1.5,
    )
    fig.supxlabel(
        "Minority - Majority mean difference",
        fontsize=8.5,
        color=COL_INK,
        y=0.028,
    )
    fig.subplots_adjust(left=0.145, right=0.99, bottom=0.145, top=0.72, wspace=0.25)
    files = save_figure(
        fig,
        output,
        "figure_outcome_selection_robustness_semantic_anatomy_style",
        dpi=300,
    )
    plt.close(fig)
    return files


def plot_main_figure(pooled: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(1, 3, figsize=(7.50, 3.65), facecolor=COL_BG)
    for index, (ax, analysis) in enumerate(zip(axes, ANALYSES[:3])):
        draw_panel(ax, pooled, analysis, show_ylabels=index == 0)
    fig.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.53, 0.988),
        ncol=2, frameon=False, fontsize=8.2, handletextpad=0.55, columnspacing=1.5,
    )
    fig.supxlabel(
        "Minority − majority mean difference",
        fontsize=8.4,
        color=COL_INK,
        y=0.035,
    )
    fig.subplots_adjust(left=0.145, right=0.99, bottom=0.155, top=0.805, wspace=0.25)
    files = save_figure(fig, output, "figure_outcome_selection_robustness", dpi=300)
    plt.close(fig)
    return files


def plot_supplement_figure(pooled: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    fig, axes = plt.subplots(2, 2, figsize=(7.50, 6.35), facecolor=COL_BG)
    for index, (ax, analysis) in enumerate(zip(axes.flat, ANALYSES)):
        draw_panel(
            ax, pooled, analysis,
            show_ylabels=index % 2 == 0,
        )
    fig.legend(
        handles=legend_handles(), loc="upper center", bbox_to_anchor=(0.53, 0.988),
        ncol=2, frameon=False, fontsize=8.2, handletextpad=0.55, columnspacing=1.5,
    )
    fig.supxlabel(
        "Minority − majority mean difference",
        fontsize=8.4,
        color=COL_INK,
        y=0.025,
    )
    fig.subplots_adjust(
        left=0.145, right=0.99, bottom=0.095, top=0.855, hspace=0.43, wspace=0.25
    )
    files = save_figure(fig, output, "figure_S1_outcome_selection_robustness", dpi=300)
    plt.close(fig)
    return files


def table_s1(pooled_merged: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis in ANALYSES:
        pdim = pooled_merged.loc[pooled_merged["analysis"].eq(analysis)]
        ploc = locations.loc[locations["analysis"].eq(analysis)]
        rows.append(
            {
                "Measure": PANEL_LABELS[analysis],
                "Pooled dimensions same direction": (
                    f"{int(pdim['direction_consistent'].sum())}/{len(pdim)}"
                ),
                "Location × dimension same direction": (
                    f"{int(ploc['direction_consistent'].sum())}/{len(ploc)}"
                ),
                "Pooled CIs overlapping": (
                    f"{int(pdim['confidence_intervals_overlap'].sum())}/{len(pdim)}"
                ),
                "Spearman rho across dimensions": round(
                    pdim["mean_difference_primary"].corr(
                        pdim["mean_difference_robustness"], method="spearman"
                    ),
                    3,
                ),
            }
        )
    return pd.DataFrame(rows)


def table_s2(locations: pd.DataFrame) -> pd.DataFrame:
    columns = [
        "analysis", "location", "dimension",
        "n_primary", "mean_difference_focal_minus_comparison_primary",
        "ci_low_primary", "ci_high_primary",
        "n_robustness", "mean_difference_focal_minus_comparison_robustness",
        "ci_low_robustness", "ci_high_robustness",
        "direction_consistent", "confidence_intervals_overlap",
    ]
    table = locations[columns].copy()
    table["analysis"] = table["analysis"].map(PANEL_LABELS)
    dimension_rank = {value: index for index, value in enumerate(DIMENSION_ORDER)}
    analysis_rank = {value: index for index, value in enumerate(PANEL_LABELS.values())}
    location_rank = {"Dallas": 0, "New York City": 1, "Los Angeles": 2}
    table["_analysis_rank"] = table["analysis"].map(analysis_rank)
    table["_dimension_rank"] = table["dimension"].map(dimension_rank)
    table["_location_rank"] = table["location"].map(location_rank)
    table = table.sort_values(
        ["_analysis_rank", "_location_rank", "_dimension_rank"]
    ).drop(
        columns=["_analysis_rank", "_location_rank", "_dimension_rank"]
    )
    result = pd.DataFrame(
        {
            "Measure": table["analysis"],
            "Location": table["location"],
            "Social dimension": table["dimension"].map(DISPLAY_DIMENSIONS),
            "Primary n": table["n_primary"].astype(int),
            "Primary effect": table[
                "mean_difference_focal_minus_comparison_primary"
            ].round(3),
            "Primary 95% CI": table.apply(
                lambda row: f"[{row['ci_low_primary']:.3f}, {row['ci_high_primary']:.3f}]",
                axis=1,
            ),
            "Robustness n": table["n_robustness"].astype(int),
            "Robustness effect": table[
                "mean_difference_focal_minus_comparison_robustness"
            ].round(3),
            "Robustness 95% CI": table.apply(
                lambda row: (
                    f"[{row['ci_low_robustness']:.3f}, {row['ci_high_robustness']:.3f}]"
                ),
                axis=1,
            ),
            "Same direction": table["direction_consistent"].map({True: "Yes", False: "No"}),
            "CIs overlap": table["confidence_intervals_overlap"].map(
                {True: "Yes", False: "No"}
            ),
        }
    )
    return result


def table_s3(pooled_merged: pd.DataFrame, locations: pd.DataFrame, base: Path) -> pd.DataFrame:
    rows = []
    for record in pooled_merged.itertuples(index=False):
        rows.append(
            {
                "level": "pooled_dimension", "metric": PANEL_LABELS[record.analysis],
                "location": "Pooled", "dimension": record.dimension,
                "primary_eligible_n": record.n_outcomes_primary, "primary_possible_n": 21,
                "robustness_eligible_n": record.n_outcomes_robustness,
                "robustness_possible_n": 7,
            }
        )
    for record in locations.itertuples(index=False):
        rows.append(
            {
                "level": "location_dimension", "metric": PANEL_LABELS[record.analysis],
                "location": record.location, "dimension": record.dimension,
                "primary_eligible_n": record.n_primary, "primary_possible_n": 21,
                "robustness_eligible_n": record.n_robustness, "robustness_possible_n": 7,
            }
        )
    aggregate_sets = []
    for label, root, possible in [
        ("primary", base / "results", 21),
        ("robustness", base / "robustness_results", 7),
    ]:
        aggregate = pd.read_csv(root / "three_location_pooled_secondary/pooled_aggregate_tests.csv")
        aggregate = aggregate[["analysis", "n_outcomes"]].rename(
            columns={"n_outcomes": f"{label}_eligible_n"}
        )
        aggregate[f"{label}_possible_n"] = possible
        aggregate_sets.append(aggregate)
    aggregate = aggregate_sets[0].merge(aggregate_sets[1], on="analysis", validate="one_to_one")
    for record in aggregate.itertuples(index=False):
        rows.append(
            {
                "level": "pooled_aggregate", "metric": PANEL_LABELS[record.analysis],
                "location": "Pooled", "dimension": "Overall",
                "primary_eligible_n": record.primary_eligible_n,
                "primary_possible_n": record.primary_possible_n,
                "robustness_eligible_n": record.robustness_eligible_n,
                "robustness_possible_n": record.robustness_possible_n,
            }
        )
    table = pd.DataFrame(rows)
    table["primary_coverage"] = table["primary_eligible_n"] / table["primary_possible_n"]
    table["robustness_coverage"] = (
        table["robustness_eligible_n"] / table["robustness_possible_n"]
    )
    level_rank = {"pooled_aggregate": 0, "pooled_dimension": 1, "location_dimension": 2}
    metric_rank = {value: index for index, value in enumerate(PANEL_LABELS.values())}
    location_rank = {"Pooled": 0, "Dallas": 1, "New York City": 2, "Los Angeles": 3}
    dimension_rank = {"Overall": -1, **{v: i for i, v in enumerate(DIMENSION_ORDER)}}
    table["_level_rank"] = table["level"].map(level_rank)
    table["_metric_rank"] = table["metric"].map(metric_rank)
    table["_location_rank"] = table["location"].map(location_rank)
    table["_dimension_rank"] = table["dimension"].map(dimension_rank)
    table = table.sort_values(
        ["_level_rank", "_metric_rank", "_location_rank", "_dimension_rank"]
    ).drop(columns=["_level_rank", "_metric_rank", "_location_rank", "_dimension_rank"])
    table = table.rename(
        columns={
            "level": "Level",
            "metric": "Measure",
            "location": "Location",
            "dimension": "Social dimension",
            "primary_eligible_n": "Primary eligible n",
            "primary_possible_n": "Primary possible n",
            "robustness_eligible_n": "Robustness eligible n",
            "robustness_possible_n": "Robustness possible n",
            "primary_coverage": "Primary coverage",
            "robustness_coverage": "Robustness coverage",
        }
    )
    table["Social dimension"] = table["Social dimension"].replace(DISPLAY_DIMENSIONS)
    table["Primary coverage"] = table["Primary coverage"].round(3)
    table["Robustness coverage"] = table["Robustness coverage"].round(3)
    return table


def save_table(table: pd.DataFrame, output: Path, stem: str) -> None:
    table.to_csv(output / f"{stem}.csv", index=False)
    latex = table.to_latex(
        index=False, longtable=True, escape=True,
        float_format=lambda value: f"{value:.3f}",
    )
    (output / f"{stem}.tex").write_text(latex, encoding="utf-8")


def write_output_readme(output: Path) -> None:
    text = """# Outcome-selection robustness: paper outputs

The main figure compares the 21 primary outcomes with the seven additional robustness
outcomes for citation count, source URL displacement, and answer semantic displacement.
The supplementary Figure S1 adds evidence semantic gap. Figures contain no overall title or
caption; those should be supplied in LaTeX.

For Figure S1, the evidence-gap panel has 11–14 eligible primary outcomes and 3–4 eligible
robustness outcomes across social dimensions. The pooled aggregate uses 18/21 primary outcomes
and 3/7 robustness outcomes. These counts belong in the caption rather than inside the panel.

Colors and typography follow `code/shared/figure_common.py`. Blue circles
represent the primary outcome set and orange diamonds represent the robustness outcome set.
Horizontal lines are 95% paired-outcome bootstrap confidence intervals.

Files:

- `figure_outcome_selection_robustness.{pdf,png}`: three-panel main-paper figure.
- `figure_outcome_selection_robustness_semantic_anatomy_style.{pdf,png}`: alternative
  three-panel version following the directional visual grammar of Figure 4.3.
- `figure_S1_outcome_selection_robustness.{pdf,png}`: four-panel supplementary figure.
- `table_S1_robustness_agreement.{csv,tex}`: directional and CI agreement summary.
- `table_S2_location_dimension_effects.{csv,tex}`: all metric × location × dimension effects.
- `table_S3_eligible_outcomes.{csv,tex}`: exact eligible and possible outcome counts.
- `figure_outcome_selection_robustness_stats.csv`: all values plotted in both figures.

Regenerate from the repository root with:

```bash
python run.py supplement
```
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_captions(output: Path) -> None:
    text = r"""% Suggested captions; edit figure numbering to match the manuscript.
\newcommand{\OutcomeRobustnessCaption}{%
Comparison of minority--majority mean differences across the 21 primary outcomes and seven
additional robustness outcomes. Points show pooled mean differences and horizontal lines show
95\% paired-outcome bootstrap confidence intervals.}

\newcommand{\OutcomeRobustnessSupplementCaption}{%
Outcome-selection robustness including evidence semantic gap. Points show pooled
minority--majority mean differences and horizontal lines show 95\% paired-outcome bootstrap
confidence intervals. Evidence-gap estimates include 11--14 eligible primary outcomes and 3--4
eligible robustness outcomes across social dimensions; exact dimension- and location-specific
counts are reported in Table~S3.}
"""
    (output / "figure_captions.tex").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = args.base_dir.absolute()
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)

    pooled = load_pooled(base)
    pooled_merged = merge_pooled_sets(pooled)
    locations = load_location_comparison(base)

    main_png, main_pdf = plot_main_figure(pooled, output)
    anatomy_png, anatomy_pdf = plot_anatomy_style_figure(pooled, output)
    supp_png, supp_pdf = plot_supplement_figure(pooled, output)
    pooled.to_csv(output / "figure_outcome_selection_robustness_stats.csv", index=False)
    save_table(table_s1(pooled_merged, locations), output, "table_S1_robustness_agreement")
    save_table(table_s2(locations), output, "table_S2_location_dimension_effects")
    save_table(
        table_s3(pooled_merged, locations, base), output, "table_S3_eligible_outcomes"
    )
    write_output_readme(output)
    write_captions(output)

    print(f"Main PNG : {main_png}")
    print(f"Main PDF : {main_pdf}")
    print(f"Alt PNG  : {anatomy_png}")
    print(f"Alt PDF  : {anatomy_pdf}")
    print(f"S1 PNG   : {supp_png}")
    print(f"S1 PDF   : {supp_pdf}")
    for name in [
        "table_S1_robustness_agreement",
        "table_S2_location_dimension_effects",
        "table_S3_eligible_outcomes",
    ]:
        print(f"Table    : {output / (name + '.csv')}")


if __name__ == "__main__":
    main()
