#!/usr/bin/env python3
"""Outcome-level heterogeneity and influence diagnostics for the primary outcomes.

For each measure, social dimension, and outcome, the script computes the
minority-minus-majority contrast within location and then averages the three
location-specific contrasts within outcome.  The resulting 21 pooled outcome
effects per dimension are summarized descriptively and subjected to a
leave-one-outcome-out influence analysis.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
from matplotlib.lines import Line2D
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/figures"))

import paper_style as ps  # noqa: E402
from pooled_stats import (  # noqa: E402
    DIMENSIONS,
    paired_location_effects,
    read_three_location_csv,
)


ZERO_TOLERANCE = 1e-12
METRICS = {
    "citation_count": {
        "label": "Citation count",
        "analysis": "link_count_analysis",
        "filename": "clean_aio_link_data.csv",
        "column": "n_links",
        "cache": "figure_4_2_pooled_volume_tests.csv",
    },
    "source_url_displacement": {
        "label": "Source URL displacement",
        "analysis": "source_overlap_analysis",
        "filename": "source_overlap_vs_generic.csv",
        "column": "url_jaccard_distance",
        "cache": "figure_4_2_pooled_source_tests.csv",
    },
    "answer_semantic_displacement": {
        "label": "Answer semantic displacement",
        "analysis": "semantic_embedding_analysis",
        "filename": "group_vs_generic_semantic_metrics.csv",
        "column": "dense_distance_subject_normalized",
        "cache": "figure_4_3_pooled_full_tests.csv",
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "supplementary_results/outcome_effect_heterogeneity",
    )
    return parser.parse_args()


def outcome_metadata(data: pd.DataFrame) -> pd.DataFrame:
    required = ["outcome_id", "domain", "outcome"]
    missing = sorted(set(required) - set(data.columns))
    if missing:
        raise RuntimeError(f"Missing outcome metadata columns: {missing}")
    metadata = data[required].drop_duplicates()
    if metadata["outcome_id"].duplicated().any():
        raise RuntimeError("An outcome_id maps to more than one domain/outcome label")
    return metadata


def build_outcome_effects(base: Path) -> pd.DataFrame:
    frames: list[pd.DataFrame] = []
    for metric, spec in METRICS.items():
        raw = read_three_location_csv(base, spec["analysis"], spec["filename"])
        metadata = outcome_metadata(raw)
        location_effects = paired_location_effects(raw, spec["column"])
        counts = (
            location_effects.groupby(["dimension", "outcome_id"])["location"]
            .nunique()
        )
        if not counts.eq(3).all():
            bad = counts.loc[~counts.eq(3)]
            raise RuntimeError(
                f"{spec['label']}: incomplete location repetitions: {bad.to_dict()}"
            )
        pooled = (
            location_effects.groupby(["dimension", "outcome_id"], as_index=False)
            .agg(
                outcome_effect=("difference", "mean"),
                n_locations=("location", "nunique"),
            )
            .merge(metadata, on="outcome_id", how="left", validate="many_to_one")
        )
        pooled.insert(0, "measure", spec["label"])
        pooled.insert(0, "measure_id", metric)
        frames.append(pooled)

    effects = pd.concat(frames, ignore_index=True)
    expected = len(METRICS) * len(DIMENSIONS) * 21
    if len(effects) != expected:
        raise RuntimeError(f"Expected {expected} pooled outcome effects; found {len(effects)}")
    cell_sizes = effects.groupby(["measure_id", "dimension"]).size()
    if not cell_sizes.eq(21).all():
        raise RuntimeError(f"Expected 21 outcomes per measure/dimension: {cell_sizes.to_dict()}")
    if effects["outcome_effect"].isna().any():
        raise RuntimeError("At least one pooled outcome effect is missing")
    return effects


def validate_against_paper_outputs(base: Path, effects: pd.DataFrame) -> None:
    cache_dir = base / "figures_three_locations_pooled"
    for metric, spec in METRICS.items():
        cached = pd.read_csv(cache_dir / spec["cache"])[["dimension", "mean_difference"]]
        reconstructed = (
            effects.loc[effects["measure_id"].eq(metric)]
            .groupby("dimension", as_index=False)["outcome_effect"]
            .mean()
            .rename(columns={"outcome_effect": "reconstructed"})
        )
        check = cached.merge(reconstructed, on="dimension", validate="one_to_one")
        discrepancy = (check["mean_difference"] - check["reconstructed"]).abs().max()
        if discrepancy > 1e-12:
            raise RuntimeError(
                f"{spec['label']}: reconstructed means do not match the paper output "
                f"(maximum discrepancy {discrepancy})"
            )


def classify_sign(values: pd.Series) -> pd.Series:
    array = values.to_numpy(float)
    signs = np.where(
        array > ZERO_TOLERANCE,
        "positive",
        np.where(array < -ZERO_TOLERANCE, "negative", "zero"),
    )
    return pd.Series(signs, index=values.index)


def summarize_distributions(effects: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for (measure_id, measure, dimension), group in effects.groupby(
        ["measure_id", "measure", "dimension"], sort=False
    ):
        values = group["outcome_effect"].astype(float)
        signs = classify_sign(values)
        q1 = float(values.quantile(0.25, interpolation="linear"))
        q3 = float(values.quantile(0.75, interpolation="linear"))
        n_positive = int(signs.eq("positive").sum())
        n_zero = int(signs.eq("zero").sum())
        n_negative = int(signs.eq("negative").sum())
        rows.append(
            {
                "measure_id": measure_id,
                "measure": measure,
                "dimension": dimension,
                "n_outcomes": int(len(values)),
                "mean": float(values.mean()),
                "median": float(values.median()),
                "q1": q1,
                "q3": q3,
                "iqr": q3 - q1,
                "n_positive": n_positive,
                "n_zero": n_zero,
                "n_negative": n_negative,
                "proportion_positive": n_positive / len(values),
            }
        )
    summary = pd.DataFrame(rows)
    summary["measure_id"] = pd.Categorical(
        summary["measure_id"], categories=list(METRICS), ordered=True
    )
    summary["dimension"] = pd.Categorical(
        summary["dimension"], categories=DIMENSIONS, ordered=True
    )
    return summary.sort_values(["measure_id", "dimension"]).reset_index(drop=True)


def leave_one_outcome_out(
    effects: pd.DataFrame,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    details = []
    summaries = []
    for (measure_id, measure, dimension), group in effects.groupby(
        ["measure_id", "measure", "dimension"], sort=False
    ):
        group = group.reset_index(drop=True)
        values = group["outcome_effect"].to_numpy(float)
        full_mean = float(values.mean())
        cell_rows = []
        for index, omitted in group.iterrows():
            loo_mean = float(np.delete(values, index).mean())
            sign_reversal = bool(full_mean * loo_mean < 0)
            row = {
                "measure_id": measure_id,
                "measure": measure,
                "dimension": dimension,
                "omitted_outcome_id": omitted["outcome_id"],
                "omitted_domain": omitted["domain"],
                "omitted_outcome": omitted["outcome"],
                "omitted_outcome_effect": float(omitted["outcome_effect"]),
                "full_mean": full_mean,
                "leave_one_out_mean": loo_mean,
                "change_from_full_mean": loo_mean - full_mean,
                "absolute_change": abs(loo_mean - full_mean),
                "sign_reversal": sign_reversal,
            }
            details.append(row)
            cell_rows.append(row)

        cell = pd.DataFrame(cell_rows)
        most_influential = cell.loc[cell["absolute_change"].idxmax()]
        summaries.append(
            {
                "measure_id": measure_id,
                "measure": measure,
                "dimension": dimension,
                "full_mean": full_mean,
                "min_leave_one_out_mean": float(cell["leave_one_out_mean"].min()),
                "max_leave_one_out_mean": float(cell["leave_one_out_mean"].max()),
                "n_sign_reversals": int(cell["sign_reversal"].sum()),
                "maximum_absolute_change": float(cell["absolute_change"].max()),
                "most_influential_outcome_id": most_influential["omitted_outcome_id"],
                "most_influential_outcome": most_influential["omitted_outcome"],
                "most_influential_domain": most_influential["omitted_domain"],
            }
        )

    detail_frame = pd.DataFrame(details)
    summary_frame = pd.DataFrame(summaries)
    for frame in (detail_frame, summary_frame):
        frame["measure_id"] = pd.Categorical(
            frame["measure_id"], categories=list(METRICS), ordered=True
        )
        frame["dimension"] = pd.Categorical(
            frame["dimension"], categories=DIMENSIONS, ordered=True
        )
        frame.sort_values(["measure_id", "dimension"], inplace=True)
        frame.reset_index(drop=True, inplace=True)
    return detail_frame, summary_frame


def distribution_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["Positive / zero / negative"] = (
        table["n_positive"].astype(str)
        + " / "
        + table["n_zero"].astype(str)
        + " / "
        + table["n_negative"].astype(str)
    )
    table["Positive proportion"] = table["proportion_positive"].map(lambda x: f"{x:.2f}")
    table["dimension"] = table["dimension"].map(ps.DIMENSION_LABELS)
    table = table.rename(
        columns={
            "measure": "Measure",
            "dimension": "Social dimension",
            "mean": "Mean",
            "median": "Median",
            "q1": "Q1",
            "q3": "Q3",
            "iqr": "IQR",
        }
    )
    return table[
        [
            "Measure",
            "Social dimension",
            "Mean",
            "Median",
            "Q1",
            "Q3",
            "IQR",
            "Positive / zero / negative",
            "Positive proportion",
        ]
    ]


def influence_paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["dimension"] = table["dimension"].map(ps.DIMENSION_LABELS)
    table = table.rename(
        columns={
            "measure": "Measure",
            "dimension": "Social dimension",
            "full_mean": "Full mean",
            "min_leave_one_out_mean": "Minimum LOO mean",
            "max_leave_one_out_mean": "Maximum LOO mean",
            "n_sign_reversals": "Sign reversals",
            "maximum_absolute_change": "Maximum absolute change",
        }
    )
    return table[
        [
            "Measure",
            "Social dimension",
            "Full mean",
            "Minimum LOO mean",
            "Maximum LOO mean",
            "Sign reversals",
            "Maximum absolute change",
        ]
    ]


def plot_distributions(
    effects: pd.DataFrame, summary: pd.DataFrame, output: Path
) -> tuple[Path, Path]:
    ps.use_paper_style()
    metric_order = list(METRICS)
    fig, axes = plt.subplots(
        1,
        3,
        figsize=(getattr(ps, "DOUBLE_COLUMN_WIDTH", 7.0), 3.25),
        sharey=True,
    )
    y_positions = np.arange(len(DIMENSIONS))[::-1]
    jitter = np.linspace(-0.16, 0.16, 21)

    for ax_index, (ax, metric) in enumerate(zip(axes, metric_order)):
        spec = METRICS[metric]
        metric_effects = effects.loc[effects["measure_id"].eq(metric)]
        metric_summary = summary.loc[summary["measure_id"].eq(metric)].set_index("dimension")
        maximum = float(metric_effects["outcome_effect"].abs().max())
        padding = max(maximum * 0.10, 0.02 if metric == "citation_count" else 0.002)

        ax.axvline(0, color=ps.INK, linewidth=0.7, zorder=0)
        for y, dimension in zip(y_positions, DIMENSIONS):
            group = (
                metric_effects.loc[metric_effects["dimension"].eq(dimension)]
                .sort_values("outcome_id")
                .reset_index(drop=True)
            )
            values = group["outcome_effect"].to_numpy(float)
            colors = np.where(
                values > ZERO_TOLERANCE,
                ps.FOCAL,
                np.where(values < -ZERO_TOLERANCE, ps.COMPARISON, ps.MUTED),
            )
            row = metric_summary.loc[dimension]
            ax.hlines(
                y,
                row["q1"],
                row["q3"],
                color=ps.RULE,
                linewidth=4.5,
                zorder=1,
            )
            ax.scatter(
                values,
                y + jitter,
                s=9,
                c=colors,
                alpha=0.72,
                edgecolors="white",
                linewidths=0.25,
                zorder=2,
            )
            ax.scatter(
                row["median"],
                y,
                marker="D",
                s=24,
                facecolor=ps.INK,
                edgecolor="white",
                linewidth=0.45,
                zorder=4,
            )
            ax.scatter(
                row["mean"],
                y,
                marker="o",
                s=27,
                facecolor="white",
                edgecolor=ps.INK,
                linewidth=0.8,
                zorder=5,
            )

        ax.set_xlim(
            float(metric_effects["outcome_effect"].min()) - padding,
            float(metric_effects["outcome_effect"].max()) + padding,
        )
        ax.set_ylim(-0.55, len(DIMENSIONS) - 0.45)
        ax.set_title(spec["label"], fontweight="bold", color=ps.INK, pad=6)
        ax.grid(axis="x", color=ps.GRID, linewidth=0.45, zorder=-1)
        ax.tick_params(axis="y", length=0)
        ps.strip_axes(ax, keep=("bottom",))
        if ax_index == 0:
            ax.set_yticks(y_positions)
            ax.set_yticklabels([ps.DIMENSION_LABELS[d] for d in DIMENSIONS])

    legend = [
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=ps.FOCAL, markeredgecolor="white", label="Positive outcome effect"),
        Line2D([], [], marker="o", linestyle="none", markersize=4.5,
               markerfacecolor=ps.COMPARISON, markeredgecolor="white", label="Negative outcome effect"),
        Line2D([], [], marker="D", linestyle="none", markersize=4.5,
               markerfacecolor=ps.INK, markeredgecolor="white", label="Median"),
        Line2D([], [], marker="o", linestyle="none", markersize=5,
               markerfacecolor="white", markeredgecolor=ps.INK, label="Mean"),
    ]
    fig.legend(
        handles=legend,
        loc="upper center",
        bbox_to_anchor=(0.5, 0.995),
        ncol=4,
        frameon=False,
        fontsize=ps.FS_NOTE,
        handletextpad=0.35,
        columnspacing=1.1,
    )
    fig.supxlabel("Minority − majority outcome-level difference", y=0.02)
    fig.subplots_adjust(left=0.18, right=0.99, top=0.84, bottom=0.18, wspace=0.16)

    png = output / "figure_S4_outcome_effect_distributions.png"
    pdf = output / "figure_S4_outcome_effect_distributions.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=ps.WHITE)
    fig.savefig(pdf, bbox_inches="tight", facecolor=ps.WHITE)
    plt.close(fig)
    return png, pdf


def write_documentation(
    output: Path,
    distributions: pd.DataFrame,
    influence: pd.DataFrame,
) -> None:
    n_positive_medians = int((distributions["median"] > ZERO_TOLERANCE).sum())
    n_majority_positive = int((distributions["proportion_positive"] > 0.5).sum())
    n_reversal_cells = int((influence["n_sign_reversals"] > 0).sum())
    total_reversals = int(influence["n_sign_reversals"].sum())
    by_measure = (
        distributions.assign(
            positive_median=distributions["median"] > ZERO_TOLERANCE,
            majority_positive=distributions["proportion_positive"] > 0.5,
        )
        .groupby("measure", observed=True)
        .agg(
            positive_medians=("positive_median", "sum"),
            majority_positive=("majority_positive", "sum"),
        )
    )
    citation = by_measure.loc[METRICS["citation_count"]["label"]]
    source = by_measure.loc[METRICS["source_url_displacement"]["label"]]
    answer = by_measure.loc[METRICS["answer_semantic_displacement"]["label"]]
    readme = f"""# Outcome-level effect heterogeneity

This analysis evaluates whether the pooled minority-minus-majority effects in the primary
21-outcome experiment are broadly distributed across outcomes or disproportionately influenced by
a small number of cases. It covers citation count, source URL displacement, and answer semantic
displacement.

For every measure, social dimension, and outcome, the contrast is first calculated separately in
Dallas, New York, and Los Angeles. The three location-specific contrasts are then averaged within
the same outcome. Locations are therefore repetitions, not independent outcomes. Every
measure-by-dimension distribution contains 21 equally weighted primary outcomes.

Across the 18 measure-by-dimension cells, {n_positive_medians}/18 medians are positive and
{n_majority_positive}/18 have more than half of their outcomes above zero. The leave-one-outcome
analysis found at least one sign reversal in {n_reversal_cells}/18 cells ({total_reversals} of 378
omissions in total).

The distribution differs by measure. Citation count has positive medians and a majority of
positive outcomes in {int(citation['positive_medians'])}/6 dimensions; answer semantic displacement
does so in {int(answer['positive_medians'])}/6. Source URL displacement has a positive median in
{int(source['positive_medians'])}/6 dimensions and a majority of positive outcomes in
{int(source['majority_positive'])}/6, reflecting the large number of exact zero contrasts visible
in Figure S4.

`positive`, `zero`, and `negative` use a numerical zero tolerance of {ZERO_TOLERANCE:g}. Quartiles
use linear interpolation. This is a descriptive heterogeneity and influence analysis; it does not
introduce a new family of hypothesis tests.

Outputs:

- `outcome_effects_long.csv`: all 378 pooled outcome effects.
- `table_S6_outcome_heterogeneity.csv` and `.tex`: distribution summaries.
- `leave_one_outcome_out_all_results.csv`: all 378 influence diagnostics, including outcome names.
- `table_S7_leave_one_outcome_influence.csv` and `.tex`: compact influence summaries.
- `figure_S4_outcome_effect_distributions.pdf` and `.png`: outcome points, IQRs, medians, and means.
- `analysis_summary.json`: machine-readable headline diagnostics.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    if n_reversal_cells == 0:
        influence_sentence = (
            "No single-outcome omission reversed the sign of any of the 18 pooled "
            "dimension-by-measure means."
        )
    else:
        influence_sentence = (
            f"At least one single-outcome omission reversed the pooled mean in "
            f"{n_reversal_cells} of the 18 dimension-by-measure combinations."
        )
    manuscript = f"""# Suggested manuscript text

We examined the distribution of the pooled minority-minus-majority contrasts across the 21
primary outcomes. Location-specific contrasts were averaged within outcome before computing the
median, interquartile range, and sign distribution for each social dimension. Across the 18
dimension-by-measure combinations, {n_positive_medians}/18 medians were positive and
{n_majority_positive}/18 contained a majority of positive outcome-level contrasts.
Citation count and answer semantic displacement met both criteria in all six dimensions, whereas
source URL displacement did so in three of six dimensions because many outcome-level contrasts
were exactly zero.
{influence_sentence} These descriptive diagnostics distinguish broadly distributed effects from
aggregate effects driven by a small number of outcomes (Tables S6--S7; Fig. S4).

Suggested Figure S4 caption: Distribution of three-location pooled minority-minus-majority effects
across the 21 primary outcomes. Each small point represents one outcome after averaging its
location-specific contrast across Dallas, New York, and Los Angeles. Thick horizontal segments
show the interquartile range, diamonds show medians, and open circles show means. Positive values
indicate larger values for the minority-marked condition.

Suggested Table S6 note: Direction counts are reported as positive / zero / negative. Positive
proportions use all eligible outcomes as the denominator, including exact zeros.
"""
    (output / "MANUSCRIPT_TEXT.md").write_text(manuscript, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = args.base_dir.absolute()
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)

    effects = build_outcome_effects(base)
    validate_against_paper_outputs(base, effects)
    distributions = summarize_distributions(effects)
    loo_details, loo_summary = leave_one_outcome_out(effects)

    effects.to_csv(output / "outcome_effects_long.csv", index=False)
    distributions.to_csv(output / "outcome_heterogeneity_full_summary.csv", index=False)
    loo_details.to_csv(output / "leave_one_outcome_out_all_results.csv", index=False)
    loo_summary.to_csv(output / "leave_one_outcome_influence_full_summary.csv", index=False)

    table_s6 = distribution_paper_table(distributions)
    table_s6.to_csv(output / "table_S6_outcome_heterogeneity.csv", index=False)
    (output / "table_S6_outcome_heterogeneity.tex").write_text(
        table_s6.to_latex(
            index=False,
            longtable=True,
            escape=True,
            float_format=lambda value: f"{value:.3f}",
        ),
        encoding="utf-8",
    )

    table_s7 = influence_paper_table(loo_summary)
    table_s7.to_csv(output / "table_S7_leave_one_outcome_influence.csv", index=False)
    (output / "table_S7_leave_one_outcome_influence.tex").write_text(
        table_s7.to_latex(
            index=False,
            longtable=True,
            escape=True,
            float_format=lambda value: f"{value:.3f}",
        ),
        encoding="utf-8",
    )

    png, pdf = plot_distributions(effects, distributions, output)
    write_documentation(output, distributions, loo_summary)

    summary = {
        "n_primary_outcomes": 21,
        "n_locations": 3,
        "n_measures": len(METRICS),
        "n_dimensions": len(DIMENSIONS),
        "n_measure_dimension_cells": int(len(distributions)),
        "n_outcome_effects": int(len(effects)),
        "positive_medians": int((distributions["median"] > ZERO_TOLERANCE).sum()),
        "cells_with_majority_positive_outcomes": int(
            (distributions["proportion_positive"] > 0.5).sum()
        ),
        "cells_with_any_leave_one_outcome_sign_reversal": int(
            (loo_summary["n_sign_reversals"] > 0).sum()
        ),
        "total_leave_one_outcome_sign_reversals": int(
            loo_summary["n_sign_reversals"].sum()
        ),
        "zero_tolerance": ZERO_TOLERANCE,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Outcome effects : {output / 'outcome_effects_long.csv'}")
    print(f"Table S6       : {output / 'table_S6_outcome_heterogeneity.csv'}")
    print(f"Table S7       : {output / 'table_S7_leave_one_outcome_influence.csv'}")
    print(f"Figure S4      : {png}")
    print(f"                 {pdf}")


if __name__ == "__main__":
    main()
