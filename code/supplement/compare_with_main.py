#!/usr/bin/env python3
"""Compare the seven-outcome robustness run with the 21-outcome main run."""

from __future__ import annotations

import argparse
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import spearmanr


ANALYSIS_LABELS = {
    "citation_count": "Citation count",
    "source_url_jaccard_distance": "Source URL displacement",
    "answer_semantic_displacement": "Answer semantic displacement",
    "evidence_semantic_gap": "Evidence semantic gap",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--main-results", type=Path, default=Path("results"))
    parser.add_argument("--robustness-results", type=Path, default=Path("robustness_results"))
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("robustness_results/comparison_with_main"),
    )
    return parser.parse_args()


def sign_consistent(left: pd.Series, right: pd.Series) -> pd.Series:
    return np.sign(left) == np.sign(right)


def ci_excludes_zero(low: pd.Series, high: pd.Series) -> pd.Series:
    return (low > 0) | (high < 0)


def ci_overlaps(
    low_a: pd.Series, high_a: pd.Series, low_b: pd.Series, high_b: pd.Series
) -> pd.Series:
    return np.maximum(low_a, low_b) <= np.minimum(high_a, high_b)


def pooled_comparison(main_root: Path, robust_root: Path, level: str) -> pd.DataFrame:
    filename = f"pooled_{level}_tests.csv"
    left = pd.read_csv(main_root / "three_location_pooled_secondary" / filename)
    right = pd.read_csv(robust_root / "three_location_pooled_secondary" / filename)
    keys = ["analysis"] if level == "aggregate" else ["analysis", "dimension"]
    merged = left.merge(right, on=keys, suffixes=("_main", "_robust"), validate="one_to_one")
    merged["absolute_change_robust_minus_main"] = (
        merged["mean_difference_robust"] - merged["mean_difference_main"]
    )
    merged["magnitude_ratio_robust_to_main"] = (
        merged["mean_difference_robust"].abs()
        / merged["mean_difference_main"].abs().replace(0, np.nan)
    )
    merged["direction_consistent"] = sign_consistent(
        merged["mean_difference_main"], merged["mean_difference_robust"]
    )
    merged["main_ci_excludes_zero"] = ci_excludes_zero(
        merged["bootstrap_ci_low_main"], merged["bootstrap_ci_high_main"]
    )
    merged["robust_ci_excludes_zero"] = ci_excludes_zero(
        merged["bootstrap_ci_low_robust"], merged["bootstrap_ci_high_robust"]
    )
    merged["confidence_intervals_overlap"] = ci_overlaps(
        merged["bootstrap_ci_low_main"],
        merged["bootstrap_ci_high_main"],
        merged["bootstrap_ci_low_robust"],
        merged["bootstrap_ci_high_robust"],
    )
    return merged


def location_comparison(main_root: Path, robust_root: Path) -> pd.DataFrame:
    relative = Path("three_location_summary/primary_dimension_effects_all_locations.csv")
    left = pd.read_csv(main_root / relative)
    right = pd.read_csv(robust_root / relative)
    keys = ["analysis", "location", "dimension"]
    merged = left.merge(right, on=keys, suffixes=("_main", "_robust"), validate="one_to_one")
    effect = "mean_difference_focal_minus_comparison"
    merged["absolute_change_robust_minus_main"] = merged[f"{effect}_robust"] - merged[f"{effect}_main"]
    merged["direction_consistent"] = sign_consistent(
        merged[f"{effect}_main"], merged[f"{effect}_robust"]
    )
    merged["main_ci_excludes_zero"] = ci_excludes_zero(merged["ci_low_main"], merged["ci_high_main"])
    merged["robust_ci_excludes_zero"] = ci_excludes_zero(
        merged["ci_low_robust"], merged["ci_high_robust"]
    )
    merged["confidence_intervals_overlap"] = ci_overlaps(
        merged["ci_low_main"],
        merged["ci_high_main"],
        merged["ci_low_robust"],
        merged["ci_high_robust"],
    )
    return merged


def build_summary(dimensions: pd.DataFrame, locations: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for analysis in ANALYSIS_LABELS:
        dim = dimensions[dimensions["analysis"] == analysis]
        loc = locations[locations["analysis"] == analysis]
        rho = spearmanr(dim["mean_difference_main"], dim["mean_difference_robust"]).statistic
        rows.append(
            {
                "analysis": analysis,
                "pooled_dimensions": len(dim),
                "pooled_direction_consistent_n": int(dim["direction_consistent"].sum()),
                "pooled_main_ci_excludes_zero_n": int(dim["main_ci_excludes_zero"].sum()),
                "pooled_robust_ci_excludes_zero_n": int(dim["robust_ci_excludes_zero"].sum()),
                "pooled_ci_overlap_n": int(dim["confidence_intervals_overlap"].sum()),
                "dimension_effect_spearman_rho": rho,
                "location_dimension_cells": len(loc),
                "location_direction_consistent_n": int(loc["direction_consistent"].sum()),
                "location_ci_overlap_n": int(loc["confidence_intervals_overlap"].sum()),
            }
        )
    return pd.DataFrame(rows)


def plot_pooled_dimensions(data: pd.DataFrame, output: Path) -> None:
    analyses = list(ANALYSIS_LABELS)
    dimensions = [
        "Race",
        "Ethnicity",
        "Gender",
        "Disability",
        "Sexual Orientation",
        "Gender Identity",
    ]
    fig, axes = plt.subplots(2, 2, figsize=(13, 10), constrained_layout=True)
    for ax, analysis in zip(axes.flat, analyses):
        subset = data[data["analysis"] == analysis].set_index("dimension").reindex(dimensions)
        y = np.arange(len(dimensions))
        for label, suffix, color, offset in [
            ("Main outcomes (n=21)", "main", "#2B6CB0", -0.12),
            ("Robustness outcomes (n=7)", "robust", "#D97706", 0.12),
        ]:
            estimate = subset[f"mean_difference_{suffix}"].to_numpy()
            low = subset[f"bootstrap_ci_low_{suffix}"].to_numpy()
            high = subset[f"bootstrap_ci_high_{suffix}"].to_numpy()
            ax.errorbar(
                estimate,
                y + offset,
                xerr=np.vstack([estimate - low, high - estimate]),
                fmt="o",
                color=color,
                ecolor=color,
                capsize=3,
                markersize=5,
                linewidth=1.2,
                label=label,
            )
        ax.axvline(0, color="#555555", linewidth=0.8, linestyle="--")
        ax.set_yticks(y, dimensions)
        ax.invert_yaxis()
        ax.set_title(ANALYSIS_LABELS[analysis])
        ax.set_xlabel("Minority − majority mean difference")
        ax.grid(axis="x", alpha=0.2)
    axes.flat[0].legend(frameon=False, loc="best")
    fig.suptitle("Main and robustness outcome sets: pooled effects by social dimension", fontsize=15)
    fig.savefig(output, dpi=300, bbox_inches="tight")
    plt.close(fig)


def write_readme(
    output: Path,
    aggregate: pd.DataFrame,
    summary: pd.DataFrame,
    extraction_rates: dict[str, float],
) -> None:
    aggregate_lines = []
    for row in aggregate.itertuples(index=False):
        aggregate_lines.append(
            f"| {ANALYSIS_LABELS[row.analysis]} | {row.mean_difference_main:.4f} "
            f"[{row.bootstrap_ci_low_main:.4f}, {row.bootstrap_ci_high_main:.4f}] | "
            f"{row.mean_difference_robust:.4f} [{row.bootstrap_ci_low_robust:.4f}, "
            f"{row.bootstrap_ci_high_robust:.4f}] | "
            f"{'yes' if row.direction_consistent else 'no'} |"
        )
    consistency_lines = []
    for row in summary.itertuples(index=False):
        consistency_lines.append(
            f"| {ANALYSIS_LABELS[row.analysis]} | "
            f"{row.pooled_direction_consistent_n}/{row.pooled_dimensions} | "
            f"{row.location_direction_consistent_n}/{row.location_dimension_cells} | "
            f"{row.pooled_robust_ci_excludes_zero_n}/{row.pooled_dimensions} |"
        )
    coverage = ", ".join(f"{k}: {100*v:.1f}%" for k, v in extraction_rates.items())
    text = f"""# Comparison of the main and robustness outcome sets

The main analysis contains 21 outcomes. The robustness analysis contains seven new outcomes,
one per domain, measured in the same three locations. Effects are minority minus majority.

## Aggregate pooled effects

| Analysis | Main estimate [95% CI] | Robustness estimate [95% CI] | Same direction |
|---|---:|---:|:---:|
{chr(10).join(aggregate_lines)}

All four aggregate effects retain their direction. Inferential precision is lower in the
robustness set because there are only seven outcomes. The evidence-semantic-gap aggregate
uses only three eligible outcomes and should therefore be treated as especially provisional.

## Consistency by social dimension

| Analysis | Pooled direction retained | Location-specific direction retained | Robustness CIs excluding zero |
|---|---:|---:|---:|
{chr(10).join(consistency_lines)}

The p-values and Holm decisions are not directly comparable as evidence of replication because
the robustness analysis has one third as many outcomes. Direction, magnitude, uncertainty, and
coverage are the primary comparison targets.

## Source-text coverage

Text extraction rates for unique source URLs were {coverage}. These rates are close to those in
the main set, but evidence synthesis is conditional on accessible/extractable sources. Missing
metadata is zero; inaccessible pages remain represented as failed/incomplete retrievals.

## Files

- `pooled_aggregate_comparison.csv`: aggregate main-versus-robustness effects.
- `pooled_dimension_comparison.csv`: comparisons for each social dimension.
- `location_dimension_comparison.csv`: comparisons within each location and dimension.
- `comparison_summary.csv`: compact directional and uncertainty diagnostics.
- `pooled_effect_comparison.png`: pooled forest comparison across dimensions.
"""
    output.write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    args.output_dir.mkdir(parents=True, exist_ok=True)

    aggregate = pooled_comparison(args.main_results, args.robustness_results, "aggregate")
    dimensions = pooled_comparison(args.main_results, args.robustness_results, "dimension")
    locations = location_comparison(args.main_results, args.robustness_results)
    summary = build_summary(dimensions, locations)

    aggregate.to_csv(args.output_dir / "pooled_aggregate_comparison.csv", index=False)
    dimensions.to_csv(args.output_dir / "pooled_dimension_comparison.csv", index=False)
    locations.to_csv(args.output_dir / "location_dimension_comparison.csv", index=False)
    summary.to_csv(args.output_dir / "comparison_summary.csv", index=False)
    plot_pooled_dimensions(dimensions, args.output_dir / "pooled_effect_comparison.png")

    extraction_rates = {}
    import json

    for location in ("dallas", "ny", "la"):
        path = args.robustness_results / f"source_collection_{location}" / "collection_summary.json"
        extraction_rates[location.title()] = json.loads(path.read_text(encoding="utf-8"))[
            "text_extraction_rate"
        ]
    write_readme(args.output_dir / "README.md", aggregate, summary, extraction_rates)
    print(f"Comparison written to {args.output_dir.resolve()}")


if __name__ == "__main__":
    main()
