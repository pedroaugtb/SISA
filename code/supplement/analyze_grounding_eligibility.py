#!/usr/bin/env python3
"""Audit effective N and condition-related eligibility for the grounding analysis."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/figures"))

import paper_style as ps  # noqa: E402
from pooled_stats import (  # noqa: E402
    DIMENSIONS,
    LOCATIONS,
    holm_adjust,
    test_differences,
)


METRIC = "evidence_semantic_gap_source_balanced"
COVERAGE_THRESHOLD = 0.50
MIN_AVAILABLE_SOURCES = 3
LOCATION_LABELS = {
    "dallas": "Dallas",
    "ny": "New York",
    "la": "Los Angeles",
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "supplementary_results/grounding_eligibility",
    )
    return parser.parse_args()


def load_data(base: Path) -> pd.DataFrame:
    frames = []
    for slug in LOCATIONS:
        path = (
            base
            / "results"
            / f"evidence_synthesis_analysis_{slug}_v2"
            / "query_evidence_alignment_metrics_v2.csv"
        )
        data = pd.read_csv(path)
        if len(data) != 273:
            raise RuntimeError(f"{slug}: expected 273 primary responses; found {len(data)}")
        required = {
            "query_id",
            "condition",
            "dimension",
            "domain",
            "outcome",
            "outcome_id",
            "fetch_coverage",
            "n_available_sources",
            METRIC,
        }
        missing = sorted(required - set(data.columns))
        if missing:
            raise RuntimeError(f"{slug}: missing columns {missing}")
        data = data.copy()
        data["location"] = slug
        data["location_label"] = LOCATION_LABELS[slug]
        frames.append(data)

    combined = pd.concat(frames, ignore_index=True)
    if len(combined) != 819 or combined["outcome_id"].nunique() != 21:
        raise RuntimeError("Expected 819 responses and 21 primary outcomes")
    if combined.duplicated(["location", "query_id"]).any():
        raise RuntimeError("Duplicate location-query observations found")

    combined["passes_coverage"] = combined["fetch_coverage"].ge(COVERAGE_THRESHOLD)
    combined["passes_source_count"] = combined["n_available_sources"].ge(
        MIN_AVAILABLE_SOURCES
    )
    combined["has_grounding_metric"] = combined[METRIC].notna()
    combined["grounding_eligible"] = (
        combined["passes_coverage"]
        & combined["passes_source_count"]
        & combined["has_grounding_metric"]
    )
    combined["eligibility_failure"] = "eligible"
    failure_masks = [
        (~combined["passes_coverage"], "coverage_below_50pct"),
        (~combined["passes_source_count"], "fewer_than_3_available_sources"),
        (~combined["has_grounding_metric"], "grounding_metric_missing"),
    ]
    for mask, label in failure_masks:
        existing = combined.loc[mask, "eligibility_failure"]
        combined.loc[mask, "eligibility_failure"] = np.where(
            existing.eq("eligible"), label, existing + ";" + label
        )
    return combined


def pair_inventory(data: pd.DataFrame) -> tuple[pd.DataFrame, pd.DataFrame]:
    explicit = data.loc[data["condition"].isin(["minority", "majority"])].copy()
    wide = explicit.pivot(
        index=["location", "location_label", "dimension", "outcome_id"],
        columns="condition",
        values="grounding_eligible",
    ).reset_index()
    if len(wide) != 378 or wide[["minority", "majority"]].isna().any().any():
        raise RuntimeError("Expected 378 complete designed focal-comparison pairs")
    wide["complete_pair"] = wide["minority"] & wide["majority"]

    pooled = (
        wide.groupby(["dimension", "outcome_id"], as_index=False)
        .agg(
            n_complete_locations=("complete_pair", "sum"),
            any_complete_location=("complete_pair", "any"),
        )
    )
    if len(pooled) != 126:
        raise RuntimeError("Expected 126 designed dimension-outcome pairs")
    return wide, pooled


def dimension_table(data: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    explicit = data.loc[data["condition"].isin(["minority", "majority"])].copy()
    eligible = (
        explicit.groupby(["dimension", "condition"], observed=True)["grounding_eligible"]
        .agg(["sum", "count"])
        .reset_index()
        .pivot(index="dimension", columns="condition", values=["sum", "count"])
    )
    rows = []
    for dimension in DIMENSIONS:
        pair_subset = pairs.loc[pairs["dimension"].eq(dimension)]
        minority_n = int(eligible.loc[dimension, ("sum", "minority")])
        majority_n = int(eligible.loc[dimension, ("sum", "majority")])
        minority_total = int(eligible.loc[dimension, ("count", "minority")])
        majority_total = int(eligible.loc[dimension, ("count", "majority")])
        complete = int(pair_subset["complete_pair"].sum())
        possible = int(len(pair_subset))
        rows.append(
            {
                "dimension": dimension,
                "minority_eligible_n": minority_n,
                "minority_total_n": minority_total,
                "minority_eligible_rate": minority_n / minority_total,
                "majority_eligible_n": majority_n,
                "majority_total_n": majority_total,
                "majority_eligible_rate": majority_n / majority_total,
                "complete_pairs_n": complete,
                "possible_pairs_n": possible,
                "complete_pair_rate": complete / possible,
            }
        )

    overall_pairs = int(pairs["complete_pair"].sum())
    rows.append(
        {
            "dimension": "Overall",
            "minority_eligible_n": int(
                explicit.loc[explicit["condition"].eq("minority"), "grounding_eligible"].sum()
            ),
            "minority_total_n": int(explicit["condition"].eq("minority").sum()),
            "minority_eligible_rate": float(
                explicit.loc[explicit["condition"].eq("minority"), "grounding_eligible"].mean()
            ),
            "majority_eligible_n": int(
                explicit.loc[explicit["condition"].eq("majority"), "grounding_eligible"].sum()
            ),
            "majority_total_n": int(explicit["condition"].eq("majority").sum()),
            "majority_eligible_rate": float(
                explicit.loc[explicit["condition"].eq("majority"), "grounding_eligible"].mean()
            ),
            "complete_pairs_n": overall_pairs,
            "possible_pairs_n": int(len(pairs)),
            "complete_pair_rate": overall_pairs / len(pairs),
        }
    )
    return pd.DataFrame(rows)


def location_table(data: pd.DataFrame, pairs: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for slug, label in LOCATION_LABELS.items():
        subset = data.loc[data["location"].eq(slug)]
        explicit = subset.loc[subset["condition"].isin(["minority", "majority"])]
        pair_subset = pairs.loc[pairs["location"].eq(slug)]
        rows.append(
            {
                "location": label,
                "all_responses_eligible_n": int(subset["grounding_eligible"].sum()),
                "all_responses_n": int(len(subset)),
                "all_responses_eligible_rate": float(subset["grounding_eligible"].mean()),
                "explicit_responses_eligible_n": int(explicit["grounding_eligible"].sum()),
                "explicit_responses_n": int(len(explicit)),
                "explicit_responses_eligible_rate": float(explicit["grounding_eligible"].mean()),
                "complete_pairs_n": int(pair_subset["complete_pair"].sum()),
                "possible_pairs_n": int(len(pair_subset)),
                "complete_pair_rate": float(pair_subset["complete_pair"].mean()),
            }
        )
    return pd.DataFrame(rows)


def condition_tests(data: pd.DataFrame) -> pd.DataFrame:
    explicit = data.loc[data["condition"].isin(["minority", "majority"])].copy()
    # Average repeated locations within outcome first, preserving the paper's
    # outcome-level inferential unit.
    rates = (
        explicit.groupby(["dimension", "outcome_id", "condition"], as_index=False)
        ["grounding_eligible"]
        .mean()
    )
    wide = rates.pivot(
        index=["dimension", "outcome_id"],
        columns="condition",
        values="grounding_eligible",
    ).reset_index()
    wide["difference"] = wide["minority"] - wide["majority"]

    rows = []
    for dimension in DIMENSIONS:
        subset = wide.loc[wide["dimension"].eq(dimension)]
        result = test_differences(
            subset["difference"],
            f"grounding_eligibility::{dimension}",
            primary="signflip",
        )
        result.update(
            {
                "analysis": "dimension",
                "dimension": dimension,
                "minority_rate": float(subset["minority"].mean()),
                "majority_rate": float(subset["majority"].mean()),
            }
        )
        rows.append(result)

    results = pd.DataFrame(rows)
    adjusted, rejected = holm_adjust(results["p_permutation_raw"].to_numpy(float))
    results["p_holm_across_dimensions"] = adjusted
    results["significant_holm"] = rejected

    overall = (
        explicit.groupby(["outcome_id", "condition"], as_index=False)["grounding_eligible"]
        .mean()
        .pivot(index="outcome_id", columns="condition", values="grounding_eligible")
        .reset_index()
    )
    overall["difference"] = overall["minority"] - overall["majority"]
    aggregate = test_differences(
        overall["difference"], "grounding_eligibility::overall", primary="signflip"
    )
    aggregate.update(
        {
            "analysis": "overall",
            "dimension": "Overall",
            "minority_rate": float(overall["minority"].mean()),
            "majority_rate": float(overall["majority"].mean()),
            "p_holm_across_dimensions": np.nan,
            "significant_holm": np.nan,
        }
    )
    return pd.concat([pd.DataFrame([aggregate]), results], ignore_index=True)


def criterion_diagnostics(data: pd.DataFrame) -> pd.DataFrame:
    explicit = data.loc[data["condition"].isin(["minority", "majority"])].copy()
    rows = []
    for condition in ["minority", "majority"]:
        subset = explicit.loc[explicit["condition"].eq(condition)]
        row = {
            "condition": condition,
            "n_responses": int(len(subset)),
            "mean_fetch_coverage": float(subset["fetch_coverage"].mean()),
        }
        for column in [
            "passes_coverage",
            "passes_source_count",
            "has_grounding_metric",
            "grounding_eligible",
        ]:
            row[f"{column}_n"] = int(subset[column].sum())
            row[f"{column}_rate"] = float(subset[column].mean())
        rows.append(row)
    return pd.DataFrame(rows)


def paper_table(summary: pd.DataFrame) -> pd.DataFrame:
    table = summary.copy()
    table["Dimension"] = table["dimension"].map(
        lambda value: ps.DIMENSION_LABELS.get(value, value)
    )
    table["Minority eligible"] = table.apply(
        lambda row: f"{int(row.minority_eligible_n)}/{int(row.minority_total_n)} "
        f"({100 * row.minority_eligible_rate:.1f}\\%)",
        axis=1,
    )
    table["Majority eligible"] = table.apply(
        lambda row: f"{int(row.majority_eligible_n)}/{int(row.majority_total_n)} "
        f"({100 * row.majority_eligible_rate:.1f}\\%)",
        axis=1,
    )
    table["Complete pairs"] = table.apply(
        lambda row: f"{int(row.complete_pairs_n)}/{int(row.possible_pairs_n)}",
        axis=1,
    )
    table["Complete-pair rate"] = table["complete_pair_rate"].map(
        lambda value: f"{100 * value:.1f}\\%"
    )
    return table[
        [
            "Dimension",
            "Minority eligible",
            "Majority eligible",
            "Complete pairs",
            "Complete-pair rate",
        ]
    ]


def main() -> None:
    args = parse_args()
    base = args.base_dir.absolute()
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)

    data = load_data(base)
    pairs, pooled_pairs = pair_inventory(data)
    dimension_summary = dimension_table(data, pairs)
    location_summary = location_table(data, pairs)
    tests = condition_tests(data)
    criteria = criterion_diagnostics(data)

    # Cross-check against the exact pair file used for Figure 4.5.
    cached_pairs = pd.read_csv(
        base / "figures_three_locations_pooled/figure_4_5_location_outcome_gap_effects.csv"
    )
    if int(pairs["complete_pair"].sum()) != len(cached_pairs):
        raise RuntimeError("Complete-pair count does not reproduce the Figure 4.5 input")
    pooled_n = int(pooled_pairs["any_complete_location"].sum())
    cached_pooled_n = int(
        pd.read_csv(
            base / "figures_three_locations_pooled/figure_4_5_pooled_gap_dimension_tests.csv"
        )["n_outcomes"].sum()
    )
    if pooled_n != cached_pooled_n:
        raise RuntimeError("Pooled complete-pair count does not reproduce Figure 4.5")

    data.to_csv(output / "grounding_response_eligibility.csv", index=False)
    pairs.to_csv(output / "grounding_location_outcome_pairs.csv", index=False)
    pooled_pairs.to_csv(output / "grounding_pooled_dimension_outcome_pairs.csv", index=False)
    dimension_summary.to_csv(output / "grounding_eligibility_by_dimension_full.csv", index=False)
    location_summary.to_csv(output / "grounding_eligibility_by_location.csv", index=False)
    tests.to_csv(output / "grounding_eligibility_condition_tests.csv", index=False)
    criteria.to_csv(output / "grounding_eligibility_criteria_by_condition.csv", index=False)

    table = paper_table(dimension_summary)
    table.to_csv(output / "table_S8_grounding_eligibility.csv", index=False)
    # Strings contain the intended LaTeX percent escape, so do not escape again.
    (output / "table_S8_grounding_eligibility.tex").write_text(
        table.to_latex(index=False, escape=False), encoding="utf-8"
    )

    all_eligible = int(data["grounding_eligible"].sum())
    explicit = data.loc[data["condition"].isin(["minority", "majority"])]
    explicit_eligible = int(explicit["grounding_eligible"].sum())
    control = data.loc[data["condition"].eq("control")]
    control_eligible = int(control["grounding_eligible"].sum())
    complete_pairs = int(pairs["complete_pair"].sum())
    continuous = pd.read_csv(
        base / "figures_three_locations_pooled/figure_4_5_pooled_condition_means.csv"
    )
    continuous_n = len(continuous)
    continuous_by_condition = continuous.groupby("condition").size().to_dict()
    continuous_by_locations = continuous["n_locations"].value_counts().sort_index().to_dict()
    overall_test = tests.iloc[0]
    criterion_lookup = criteria.set_index("condition")
    p_text = (
        "<.001"
        if overall_test["p_permutation_raw"] < 0.001
        else f"={overall_test['p_permutation_raw']:.3f}"
    )

    summary = {
        "eligibility_rule": {
            "minimum_fetch_coverage": COVERAGE_THRESHOLD,
            "minimum_available_sources": MIN_AVAILABLE_SOURCES,
            "metric_must_be_nonmissing": METRIC,
        },
        "all_primary_responses": len(data),
        "all_primary_eligible": all_eligible,
        "all_primary_eligible_rate": all_eligible / len(data),
        "explicit_group_responses": len(explicit),
        "explicit_group_eligible": explicit_eligible,
        "explicit_group_eligible_rate": explicit_eligible / len(explicit),
        "control_responses": len(control),
        "control_eligible": control_eligible,
        "control_eligible_rate": control_eligible / len(control),
        "complete_location_outcome_pairs": complete_pairs,
        "possible_location_outcome_pairs": len(pairs),
        "complete_location_outcome_pair_rate": complete_pairs / len(pairs),
        "pooled_dimension_outcome_pairs_with_at_least_one_complete_location": pooled_n,
        "possible_pooled_dimension_outcome_pairs": len(pooled_pairs),
        "figure_4_5_continuous_association_condition_outcome_means": continuous_n,
        "figure_4_5_condition_outcome_means_by_condition": {
            str(key): int(value) for key, value in continuous_by_condition.items()
        },
        "figure_4_5_condition_outcome_means_by_eligible_location_count": {
            str(int(key)): int(value) for key, value in continuous_by_locations.items()
        },
        "minority_eligible_rate": float(overall_test["minority_rate"]),
        "majority_eligible_rate": float(overall_test["majority_rate"]),
        "minority_minus_majority_rate_difference": float(overall_test["mean_difference"]),
        "eligibility_difference_ci": [
            float(overall_test["bootstrap_ci_low"]),
            float(overall_test["bootstrap_ci_high"]),
        ],
        "eligibility_difference_p_signflip": float(overall_test["p_permutation_raw"]),
        "criterion_pass_rates_by_condition": {
            condition: {
                "coverage_at_least_50pct": float(
                    criterion_lookup.loc[condition, "passes_coverage_rate"]
                ),
                "at_least_3_available_sources": float(
                    criterion_lookup.loc[condition, "passes_source_count_rate"]
                ),
                "grounding_metric_nonmissing": float(
                    criterion_lookup.loc[condition, "has_grounding_metric_rate"]
                ),
            }
            for condition in ["minority", "majority"]
        },
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    manuscript = f"""# Suggested manuscript text

Of the {len(data)} primary AIO responses, {all_eligible} ({100 * all_eligible / len(data):.1f}%)
met the reference evidence-recovery criterion of at least 50% of cited sources recovered, at least
three usable sources, and a nonmissing source-balanced grounding score. Among the {len(explicit)}
explicit-group responses used to construct minority--majority comparisons, {explicit_eligible}
({100 * explicit_eligible / len(explicit):.1f}%) were eligible. This yielded {complete_pairs} of
{len(pairs)} ({100 * complete_pairs / len(pairs):.1f}%) complete location-specific
minority--majority outcome pairs and {pooled_n} of {len(pooled_pairs)} pooled
dimension--outcome pairs with at least one complete location.

The continuous blocked association shown in Figure 4.5 used {continuous_n} pooled explicit
condition--outcome means across all 21 outcomes ({continuous_by_condition.get('minority', 0)}
minority and {continuous_by_condition.get('majority', 0)} majority). Of these,
{continuous_by_locations.get(1, 0)} were informed by one eligible location,
{continuous_by_locations.get(2, 0)} by two locations, and
{continuous_by_locations.get(3, 0)} by all three locations.

Eligibility was {100 * overall_test['minority_rate']:.1f}% for minority-marked responses and
{100 * overall_test['majority_rate']:.1f}% for majority-marked responses, a difference of
{100 * overall_test['mean_difference']:.1f} percentage points (95% outcome-bootstrap CI
[{100 * overall_test['bootstrap_ci_low']:.1f}, {100 * overall_test['bootstrap_ci_high']:.1f}];
paired outcome-level sign-flip $p{p_text}$).

The eligibility difference reflects multiple components of the reference rule. The 50% recovery
threshold was met by {100 * criterion_lookup.loc['minority', 'passes_coverage_rate']:.1f}% of
minority-marked and {100 * criterion_lookup.loc['majority', 'passes_coverage_rate']:.1f}% of
majority-marked responses; the minimum of three recovered sources was met by
{100 * criterion_lookup.loc['minority', 'passes_source_count_rate']:.1f}% and
{100 * criterion_lookup.loc['majority', 'passes_source_count_rate']:.1f}%, respectively. Grounding
results should therefore be interpreted as conditional on evidence-recovery eligibility rather
than representative of every collected response.

Suggested Table S8 note: Eligibility required at least 50% recovered-source coverage, at least
three usable recovered sources, and a nonmissing source-balanced evidence-alignment score.
Complete pairs require both the minority- and majority-marked response for the same outcome and
location to be eligible. The complete-pair denominator is 63 per dimension (21 outcomes across
three locations). The condition comparison averages the three location repetitions within outcome
before inference across the 21 outcomes.
"""
    (output / "MANUSCRIPT_TEXT.md").write_text(manuscript, encoding="utf-8")

    readme = f"""# Grounding eligibility and effective sample size

This audit uses the exact reference rule and primary inputs used by Figure 4.5. It distinguishes
response-level eligibility, location-specific complete focal--comparison pairs, pooled
dimension--outcome pairs, and the condition--outcome means entering the continuous grounding
association.

Headline counts:

- all primary responses: {all_eligible}/{len(data)} eligible ({100 * all_eligible / len(data):.1f}%);
- explicit minority/majority responses: {explicit_eligible}/{len(explicit)} eligible ({100 * explicit_eligible / len(explicit):.1f}%);
- control responses: {control_eligible}/{len(control)} eligible ({100 * control_eligible / len(control):.1f}%);
- complete location-specific focal--comparison pairs: {complete_pairs}/{len(pairs)} ({100 * complete_pairs / len(pairs):.1f}%);
- pooled dimension--outcome pairs with at least one complete location: {pooled_n}/{len(pooled_pairs)};
- condition--outcome means in the Figure 4.5 continuous association: {continuous_n}.

The {continuous_n} condition--outcome means comprise
{continuous_by_condition.get('minority', 0)} minority and
{continuous_by_condition.get('majority', 0)} majority observations;
{continuous_by_locations.get(1, 0)} are based on one eligible location,
{continuous_by_locations.get(2, 0)} on two, and {continuous_by_locations.get(3, 0)} on all three.

The condition comparison uses outcome, rather than response-location, as the inferential unit.
Location-specific binary eligibility indicators are averaged within outcome and condition before
the paired sign-flip test and outcome bootstrap. Dimension-specific p-values are Holm-adjusted;
the overall contrast is the prespecified summary and is reported separately.

Outputs include the paper-facing Table S8, full response and pair inventories, condition tests,
criterion-level diagnostics, `analysis_summary.json`, and suggested manuscript language.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    print(f"Table       : {output / 'table_S8_grounding_eligibility.csv'}")
    print(f"Tests       : {output / 'grounding_eligibility_condition_tests.csv'}")
    print(f"Summary     : {output / 'analysis_summary.json'}")
    print(f"Manuscript  : {output / 'MANUSCRIPT_TEXT.md'}")


if __name__ == "__main__":
    main()
