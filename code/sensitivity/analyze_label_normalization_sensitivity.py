#!/usr/bin/env python3
"""Compare raw and experimental-label-normalized answer displacement."""

from __future__ import annotations

import argparse
import json
import math
import sys
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


# Use the primary pooled implementation itself so the official normalized
# source-to-answer result and its Monte Carlo reference are exactly reproduced.
POOLED_STATS_DIR = Path(__file__).resolve().parents[1] / "figures"
sys.path.insert(0, str(POOLED_STATS_DIR))
from pooled_stats import blocked_rank_correlation as primary_blocked_rank_correlation


BASE_DEFAULT = Path(__file__).resolve().parents[2]
LOCATIONS = ("dallas", "ny", "la")
REPRESENTATIONS = {
    "normalized": "dense_distance_subject_normalized",
    "raw": "dense_distance_raw",
}
SOURCE_METRIC = "url_jaccard_distance"
SEED = 42
COLORS = {"normalized": "#3B6FB6", "raw": "#D97706"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    return parser.parse_args()


def stable_seed(label: str) -> int:
    return (SEED + zlib.crc32(label.encode("utf-8"))) % (2**32 - 1)


def read_semantic(base: Path) -> pd.DataFrame:
    frames = []
    for location in LOCATIONS:
        path = (
            base
            / "results"
            / f"semantic_embedding_analysis_{location}"
            / "group_vs_generic_semantic_metrics.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["location"] = location
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    required = {
        "location", "dimension", "condition", "group", "domain", "outcome",
        "outcome_id", *REPRESENTATIONS.values(),
    }
    missing = required.difference(data.columns)
    if missing:
        raise ValueError(f"Semantic results are missing columns: {sorted(missing)}")
    keys = ["location", "dimension", "condition", "group", "outcome_id"]
    if data.duplicated(keys).any():
        raise ValueError("Semantic rows are not unique on the planned merge keys.")
    return data


def read_source_overlap(base: Path) -> pd.DataFrame:
    frames = []
    for location in LOCATIONS:
        path = (
            base
            / "results"
            / f"source_overlap_analysis_{location}"
            / "source_overlap_vs_generic.csv"
        )
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["location"] = location
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    keys = ["location", "dimension", "condition", "group", "outcome_id"]
    if data.duplicated(keys).any():
        raise ValueError("Source-overlap rows are not unique on the planned merge keys.")
    return data


def explicit_rows(frame: pd.DataFrame) -> pd.DataFrame:
    return frame.loc[frame["condition"].isin(["minority", "majority"])].copy()


def location_pairs(frame: pd.DataFrame, metric: str) -> pd.DataFrame:
    data = explicit_rows(frame)
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    keys = ["location", "dimension", "domain", "outcome", "outcome_id"]
    pivot = data.pivot_table(index=keys, columns="condition", values=metric, aggfunc="mean")
    pivot = pivot.dropna(subset=["minority", "majority"]).reset_index()
    pivot["difference"] = pivot["minority"] - pivot["majority"]
    expected = len(LOCATIONS) * 6 * 21
    if len(pivot) != expected:
        raise ValueError(f"Expected {expected} location-level pairs; found {len(pivot)}.")
    return pivot


def pooled_outcome_pairs(location_effects: pd.DataFrame) -> pd.DataFrame:
    keys = ["dimension", "domain", "outcome", "outcome_id"]
    pooled = location_effects.groupby(keys, as_index=False).agg(
        minority=("minority", "mean"),
        majority=("majority", "mean"),
        difference=("difference", "mean"),
        n_locations=("location", "nunique"),
    )
    if len(pooled) != 6 * 21 or not pooled["n_locations"].eq(3).all():
        raise ValueError("Expected 126 complete three-location dimension-outcome effects.")
    return pooled


def bootstrap_mean_ci(values: np.ndarray, label: str, n_bootstrap: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    rng = np.random.default_rng(stable_seed("mean::" + label))
    indices = rng.integers(0, len(values), size=(n_bootstrap, len(values)))
    means = values[indices].mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def signflip_p(values: np.ndarray, label: str, n_permutations: int) -> float:
    values = values[np.isfinite(values)]
    values = values[~np.isclose(values, 0.0)]
    if not len(values):
        return 1.0
    observed = abs(float(values.mean()))
    rng = np.random.default_rng(stable_seed("signflip::" + label))
    extreme = 0
    complete = 0
    while complete < n_permutations:
        size = min(10000, n_permutations - complete)
        signs = rng.choice((-1.0, 1.0), size=(size, len(values)))
        extreme += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
        complete += size
    return float((extreme + 1) / (n_permutations + 1))


def holm_adjust(values: pd.Series) -> np.ndarray:
    p = values.to_numpy(float)
    order = np.argsort(p)
    adjusted_sorted = np.empty(len(p), dtype=float)
    running = 0.0
    for position, index in enumerate(order):
        running = max(running, min(1.0, (len(p) - position) * p[index]))
        adjusted_sorted[position] = running
    adjusted = np.empty(len(p), dtype=float)
    adjusted[order] = adjusted_sorted
    return adjusted


def dimension_table(
    pooled: pd.DataFrame, representation: str, n_bootstrap: int, n_permutations: int
) -> pd.DataFrame:
    rows = []
    for dimension, group in pooled.groupby("dimension", sort=True):
        values = group["difference"].to_numpy(float)
        low, high = bootstrap_mean_ci(values, f"{representation}::{dimension}", n_bootstrap)
        rows.append(
            {
                "representation": representation,
                "dimension": dimension,
                "n_outcomes": int(len(values)),
                "mean_difference": float(values.mean()),
                "median_difference": float(np.median(values)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "focal_mean": float(group["minority"].mean()),
                "comparison_mean": float(group["majority"].mean()),
                "p_signflip_raw": signflip_p(
                    values, f"{representation}::{dimension}", n_permutations
                ),
            }
        )
    table = pd.DataFrame(rows)
    table["p_signflip_holm"] = holm_adjust(table["p_signflip_raw"])
    table["significant_holm"] = table["p_signflip_holm"] < 0.05
    return table


def direction(value: float, tolerance: float = 1e-12) -> int:
    return int(value > tolerance) - int(value < -tolerance)


def agreement_summary(
    normalized: pd.DataFrame,
    raw: pd.DataFrame,
    keys: list[str],
    value: str,
    unit: str,
) -> tuple[pd.DataFrame, dict]:
    left = normalized[keys + [value]].rename(columns={value: "effect_normalized"})
    right = raw[keys + [value]].rename(columns={value: "effect_raw"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    x = merged["effect_normalized"].to_numpy(float)
    y = merged["effect_raw"].to_numpy(float)
    rho, p_value = spearmanr(x, y)
    merged["same_direction"] = [direction(a) == direction(b) for a, b in zip(x, y)]
    summary = {
        "unit": unit,
        "n_units": int(len(merged)),
        "same_direction_n": int(merged["same_direction"].sum()),
        "same_direction_rate": float(merged["same_direction"].mean()),
        "spearman_rho": float(rho),
        "spearman_p_descriptive": float(p_value),
        "mean_absolute_effect_change": float(np.mean(np.abs(y - x))),
        "median_absolute_effect_change": float(np.median(np.abs(y - x))),
    }
    merged.insert(0, "unit", unit)
    return merged, summary


def merge_source_and_answer(source: pd.DataFrame, semantic: pd.DataFrame) -> pd.DataFrame:
    keys = ["location", "dimension", "condition", "group", "outcome_id"]
    answer_columns = keys + list(REPRESENTATIONS.values())
    merged = explicit_rows(source).merge(
        explicit_rows(semantic)[answer_columns],
        on=keys,
        how="inner",
        validate="one_to_one",
    )
    expected = len(LOCATIONS) * 12 * 21
    if len(merged) != expected:
        raise ValueError(f"Expected {expected} merged explicit rows; found {len(merged)}.")
    if merged[[SOURCE_METRIC, *REPRESENTATIONS.values()]].isna().any().any():
        raise ValueError("Missing source or answer metric after the paired merge.")
    return merged


def pooled_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["group", "condition", "dimension", "domain", "outcome", "outcome_id"]
    numeric = [SOURCE_METRIC, *REPRESENTATIONS.values()]
    pooled = frame.groupby(keys, as_index=False).agg(
        **{column: (column, "mean") for column in numeric},
        n_locations=("location", "nunique"),
    )
    if len(pooled) != 12 * 21 or not pooled["n_locations"].eq(3).all():
        raise ValueError("Expected 252 complete three-location explicit condition means.")
    return pooled


def block_parts(
    frame: pd.DataFrame, y: str
) -> list[tuple[str, np.ndarray, np.ndarray]]:
    parts = []
    for outcome, group in frame.groupby("outcome_id", sort=False):
        xv = pd.to_numeric(group[SOURCE_METRIC], errors="coerce").to_numpy(float)
        yv = pd.to_numeric(group[y], errors="coerce").to_numpy(float)
        valid = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[valid], yv[valid]
        if len(xv) < 3:
            continue
        xr = rankdata(xv, method="average"); xr -= xr.mean()
        yr = rankdata(yv, method="average"); yr -= yr.mean()
        parts.append((str(outcome), xr, yr))
    return parts


def correlation_from_parts(parts: list[tuple[str, np.ndarray, np.ndarray]]) -> float:
    numerator = sum(float(xr @ yr) for _, xr, yr in parts)
    x2 = sum(float(xr @ xr) for _, xr, _ in parts)
    y2 = sum(float(yr @ yr) for _, _, yr in parts)
    return float(numerator / math.sqrt(x2 * y2))


def blocked_correlation(
    frame: pd.DataFrame,
    representation: str,
    n_bootstrap: int,
    n_permutations: int,
) -> tuple[dict, dict[str, tuple[str, np.ndarray, np.ndarray]]]:
    metric = REPRESENTATIONS[representation]
    parts = block_parts(frame, metric)
    if len(parts) != 21:
        raise ValueError(f"Expected 21 outcome blocks for {representation}; found {len(parts)}.")
    primary_result = primary_blocked_rank_correlation(
        frame,
        SOURCE_METRIC,
        metric,
        block="outcome_id",
        label=f"{SOURCE_METRIC}::{metric}",
        n_perm=n_permutations,
        n_boot=n_bootstrap,
    )
    if not primary_result:
        raise ValueError(f"Primary blocked-correlation implementation failed for {representation}.")
    row = {
        "representation": representation,
        "metric_x": SOURCE_METRIC,
        "metric_y": metric,
        "n_observations": int(primary_result["n_observations"]),
        "n_outcomes": int(primary_result["n_blocks"]),
        "blocked_rank_correlation": float(primary_result["blocked_rank_correlation"]),
        "cluster_bootstrap_ci_low": float(primary_result["cluster_bootstrap_ci_low"]),
        "cluster_bootstrap_ci_high": float(primary_result["cluster_bootstrap_ci_high"]),
        "p_blocked_permutation": float(primary_result["p_blocked_permutation_raw"]),
    }
    return row, {part[0]: part for part in parts}


def validate_official_normalized_reference(base: Path, correlations: pd.DataFrame) -> dict:
    path = base / "figures_three_locations_pooled" / "pooled_pipeline_correlations.csv"
    if not path.exists():
        raise FileNotFoundError(f"Official pooled correlation output is required: {path}")
    official = pd.read_csv(path)
    target = official.loc[
        official["metric_x"].eq(SOURCE_METRIC)
        & official["metric_y"].eq(REPRESENTATIONS["normalized"])
    ]
    if len(target) != 1:
        raise ValueError(f"Expected one official normalized source-to-answer row; found {len(target)}.")
    expected = target.iloc[0]
    actual = correlations.loc[correlations["representation"].eq("normalized")].iloc[0]
    comparisons = {
        "blocked_rank_correlation": "blocked_rank_correlation",
        "cluster_bootstrap_ci_low": "cluster_bootstrap_ci_low",
        "cluster_bootstrap_ci_high": "cluster_bootstrap_ci_high",
        "p_blocked_permutation": "p_blocked_permutation_raw",
    }
    for actual_column, official_column in comparisons.items():
        if not np.isclose(
            float(actual[actual_column]), float(expected[official_column]), rtol=0.0, atol=1e-15
        ):
            raise ValueError(
                f"Normalized {actual_column} does not reproduce the official pooled value: "
                f"{actual[actual_column]} != {expected[official_column]}"
            )
    return {
        "official_file": str(path),
        "exact_match": True,
        "blocked_rank_correlation": float(expected["blocked_rank_correlation"]),
        "cluster_bootstrap_ci_low": float(expected["cluster_bootstrap_ci_low"]),
        "cluster_bootstrap_ci_high": float(expected["cluster_bootstrap_ci_high"]),
        "p_blocked_permutation": float(expected["p_blocked_permutation_raw"]),
    }


def correlation_difference(
    normalized_parts: dict[str, tuple[str, np.ndarray, np.ndarray]],
    raw_parts: dict[str, tuple[str, np.ndarray, np.ndarray]],
    n_bootstrap: int,
) -> dict:
    outcomes = sorted(set(normalized_parts) & set(raw_parts))
    normalized = [normalized_parts[key] for key in outcomes]
    raw = [raw_parts[key] for key in outcomes]
    normalized_r = correlation_from_parts(normalized)
    raw_r = correlation_from_parts(raw)
    rng = np.random.default_rng(stable_seed("correlation-difference"))
    differences = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(outcomes), size=len(outcomes))
        differences[index] = correlation_from_parts([raw[i] for i in sample]) - correlation_from_parts(
            [normalized[i] for i in sample]
        )
    low, high = np.percentile(differences[np.isfinite(differences)], [2.5, 97.5])
    return {
        "n_common_outcomes": int(len(outcomes)),
        "normalized_r": normalized_r,
        "raw_r": raw_r,
        "difference_raw_minus_normalized": raw_r - normalized_r,
        "difference_ci_low": float(low),
        "difference_ci_high": float(high),
    }


def plot_effects(dimension_effects: pd.DataFrame, output: Path) -> None:
    normalized = dimension_effects.loc[
        dimension_effects["representation"] == "normalized",
        ["dimension", "mean_difference"],
    ]
    raw = dimension_effects.loc[
        dimension_effects["representation"] == "raw",
        ["dimension", "mean_difference"],
    ]
    merged = normalized.merge(raw, on="dimension", suffixes=("_normalized", "_raw"))
    x = merged["mean_difference_normalized"]
    y = merged["mean_difference_raw"]
    values = np.r_[x, y]
    pad = max(float(values.max() - values.min()) * 0.12, 0.005)
    low, high = float(values.min() - pad), float(values.max() + pad)
    fig, ax = plt.subplots(figsize=(4.4, 4.0))
    ax.scatter(x, y, s=48, color=COLORS["raw"], edgecolor="white", linewidth=0.6)
    ax.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle="--")
    ax.axhline(0, color="#777777", linewidth=0.6)
    ax.axvline(0, color="#777777", linewidth=0.6)
    for _, row in merged.iterrows():
        ax.annotate(
            row["dimension"],
            (row["mean_difference_normalized"], row["mean_difference_raw"]),
            xytext=(3, 3), textcoords="offset points", fontsize=7,
        )
    ax.set_xlim(low, high); ax.set_ylim(low, high)
    ax.set_xlabel("Label-normalized displacement effect", fontsize=9)
    ax.set_ylabel("Raw-text displacement effect", fontsize=9)
    ax.set_title("Raw vs. label-normalized effects", fontsize=10, weight="bold")
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_correlations(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(5.7, 2.45))
    positions = {"normalized": 1, "raw": 0}
    for _, row in table.iterrows():
        y = positions[row["representation"]]
        ax.errorbar(
            row["blocked_rank_correlation"], y,
            xerr=[
                [row["blocked_rank_correlation"] - row["cluster_bootstrap_ci_low"]],
                [row["cluster_bootstrap_ci_high"] - row["blocked_rank_correlation"]],
            ],
            fmt="o", color=COLORS[row["representation"]], markersize=5, capsize=2.5,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks([0, 1], ["Raw text", "Label normalized"])
    ax.set_xlabel("Outcome-blocked rank correlation", fontsize=9)
    ax.set_title("Source displacement × answer displacement", fontsize=10, weight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def manuscript_text(
    agreement: pd.DataFrame, correlations: pd.DataFrame, difference: dict
) -> str:
    pooled = agreement.loc[agreement["unit"] == "pooled_dimension"].iloc[0]
    location = agreement.loc[agreement["unit"] == "location_dimension"].iloc[0]
    outcome = agreement.loc[agreement["unit"] == "dimension_outcome"].iloc[0]
    normalized = correlations.loc[correlations["representation"] == "normalized"].iloc[0]
    raw = correlations.loc[correlations["representation"] == "raw"].iloc[0]
    return (
        "# Manuscript text: raw versus label-normalized answer displacement\n\n"
        "As a representation sensitivity check, we recomputed answer semantic "
        "displacement from embeddings of the unmodified response text rather than "
        "responses in which experimental group descriptors were replaced by a common "
        "reference. Raw and label-normalized pooled dimension effects agreed in sign "
        f"for {int(pooled['same_direction_n'])} of {int(pooled['n_units'])} dimensions "
        f"and were rank-correlated at Spearman rho = {pooled['spearman_rho']:.3f}. "
        f"Directional agreement was {int(location['same_direction_n'])}/"
        f"{int(location['n_units'])} across location-by-dimension effects and "
        f"{int(outcome['same_direction_n'])}/{int(outcome['n_units'])} across "
        "dimension-by-outcome effects. The outcome-blocked association between source "
        "URL displacement and answer displacement was similar using normalized "
        f"responses (r = {normalized['blocked_rank_correlation']:.3f}, 95% CI "
        f"[{normalized['cluster_bootstrap_ci_low']:.3f}, "
        f"{normalized['cluster_bootstrap_ci_high']:.3f}]) and raw responses "
        f"(r = {raw['blocked_rank_correlation']:.3f}, 95% CI "
        f"[{raw['cluster_bootstrap_ci_low']:.3f}, "
        f"{raw['cluster_bootstrap_ci_high']:.3f}]). The paired difference was "
        f"{difference['difference_raw_minus_normalized']:+.3f} (95% CI "
        f"[{difference['difference_ci_low']:.3f}, "
        f"{difference['difference_ci_high']:.3f}]). These results indicate that the "
        "semantic-displacement pattern was not induced by descriptor normalization.\n"
    )


def main() -> None:
    options = parse_args()
    base = options.base_dir.resolve()
    output = (
        options.output_dir.resolve()
        if options.output_dir
        else base / "sensitivity_results/label_normalization"
    )
    output.mkdir(parents=True, exist_ok=True)

    semantic = read_semantic(base)
    source = read_source_overlap(base)

    location_cache = {
        name: location_pairs(semantic, metric)
        for name, metric in REPRESENTATIONS.items()
    }
    pooled_cache = {
        name: pooled_outcome_pairs(frame)
        for name, frame in location_cache.items()
    }
    dimension_effects = pd.concat(
        [
            dimension_table(
                pooled_cache[name], name, options.n_bootstrap, options.n_permutations
            )
            for name in REPRESENTATIONS
        ],
        ignore_index=True,
    )
    dimension_effects.to_csv(output / "dimension_effects_by_representation.csv", index=False)

    normalized_dimensions = dimension_effects.loc[
        dimension_effects["representation"] == "normalized"
    ]
    raw_dimensions = dimension_effects.loc[dimension_effects["representation"] == "raw"]
    dimension_comparison = normalized_dimensions[
        ["dimension", "mean_difference", "bootstrap_ci_low", "bootstrap_ci_high"]
    ].merge(
        raw_dimensions[
            ["dimension", "mean_difference", "bootstrap_ci_low", "bootstrap_ci_high"]
        ],
        on="dimension",
        validate="one_to_one",
        suffixes=("_normalized", "_raw"),
    )
    dimension_comparison["same_direction"] = [
        direction(a) == direction(b)
        for a, b in zip(
            dimension_comparison["mean_difference_normalized"],
            dimension_comparison["mean_difference_raw"],
        )
    ]
    dimension_comparison.to_csv(output / "table_raw_vs_normalized_effects.csv", index=False)
    location_dimension = {
        name: frame.groupby(["location", "dimension"], as_index=False)["difference"].mean()
        for name, frame in location_cache.items()
    }
    details = []
    summaries = []
    detail, summary = agreement_summary(
        normalized_dimensions, raw_dimensions, ["dimension"], "mean_difference",
        "pooled_dimension",
    )
    details.append(detail); summaries.append(summary)
    detail, summary = agreement_summary(
        location_dimension["normalized"], location_dimension["raw"],
        ["location", "dimension"], "difference", "location_dimension",
    )
    details.append(detail); summaries.append(summary)
    detail, summary = agreement_summary(
        pooled_cache["normalized"], pooled_cache["raw"],
        ["dimension", "domain", "outcome", "outcome_id"], "difference",
        "dimension_outcome",
    )
    details.append(detail); summaries.append(summary)
    agreement = pd.DataFrame(summaries)
    agreement.to_csv(output / "effect_stability_summary.csv", index=False)
    pd.concat(details, ignore_index=True).to_csv(
        output / "effect_stability_details.csv", index=False
    )

    merged = merge_source_and_answer(source, semantic)
    pooled = pooled_conditions(merged)
    pooled.to_csv(output / "pooled_source_answer_metrics.csv", index=False)
    correlation_rows = []
    parts = {}
    for representation in REPRESENTATIONS:
        row, representation_parts = blocked_correlation(
            pooled, representation, options.n_bootstrap, options.n_permutations
        )
        correlation_rows.append(row)
        parts[representation] = representation_parts
    correlations = pd.DataFrame(correlation_rows)
    official_validation = validate_official_normalized_reference(base, correlations)
    correlations.to_csv(output / "blocked_correlations_by_representation.csv", index=False)
    difference = correlation_difference(
        parts["normalized"], parts["raw"], options.n_bootstrap
    )
    pd.DataFrame([difference]).to_csv(
        output / "blocked_correlation_difference.csv", index=False
    )

    sample_audit = pd.DataFrame(
        [
            {"stage": "semantic_all", "n_rows": len(semantic), "n_locations": 3},
            {"stage": "semantic_explicit", "n_rows": len(explicit_rows(semantic)), "n_locations": 3},
            {"stage": "location_dimension_outcome_pairs_per_representation", "n_rows": len(location_cache["normalized"]), "n_locations": 3},
            {"stage": "pooled_dimension_outcome_pairs_per_representation", "n_rows": len(pooled_cache["normalized"]), "n_locations": 3},
            {"stage": "pooled_source_answer_conditions", "n_rows": len(pooled), "n_locations": 3},
        ]
    )
    sample_audit.to_csv(output / "sample_audit.csv", index=False)

    plot_effects(dimension_effects, output / "figure_raw_vs_normalized_effects")
    plot_correlations(correlations, output / "figure_raw_vs_normalized_correlations")
    text = manuscript_text(agreement, correlations, difference)
    (output / "MANUSCRIPT_TEXT.md").write_text(text, encoding="utf-8")
    summary = {
        "analysis": "Raw versus experimental-label-normalized answer displacement",
        "representations": REPRESENTATIONS,
        "n_bootstrap": options.n_bootstrap,
        "n_permutations": options.n_permutations,
        "effect_stability": agreement.to_dict("records"),
        "blocked_correlations": correlations.to_dict("records"),
        "blocked_correlation_difference": difference,
        "official_normalized_validation": official_validation,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Raw versus label-normalized answer displacement\n\n"
        "This post-processing sensitivity analysis uses the already-computed raw-text "
        "and experimental-label-normalized MPNet distances. No embeddings are rerun. "
        "Focal-comparison pairs are formed within location before city repetitions are "
        "averaged within dimension and outcome. The source-to-answer analysis uses the "
        "same 252 pooled explicit-condition observations and 21 outcome blocks for both "
        "representations. Spearman tests are descriptive; substantive robustness is "
        "assessed from signs, ranks, magnitudes, and paired blocked correlations.\n\n"
        f"Pooled dimension effects agreed in sign for {int(agreement.iloc[0]['same_direction_n'])}/"
        f"{int(agreement.iloc[0]['n_units'])} dimensions (Spearman rho = "
        f"{agreement.iloc[0]['spearman_rho']:.3f}); agreement was "
        f"{int(agreement.iloc[1]['same_direction_n'])}/{int(agreement.iloc[1]['n_units'])} "
        "at location by dimension and "
        f"{int(agreement.iloc[2]['same_direction_n'])}/{int(agreement.iloc[2]['n_units'])} "
        "at dimension by outcome. The blocked source-to-answer correlation was "
        f"{correlations.iloc[0]['blocked_rank_correlation']:.3f} normalized and "
        f"{correlations.iloc[1]['blocked_rank_correlation']:.3f} raw; the paired raw-minus-"
        f"normalized difference was {difference['difference_raw_minus_normalized']:+.3f} "
        f"[{difference['difference_ci_low']:.3f}, {difference['difference_ci_high']:.3f}].\n",
        encoding="utf-8",
    )
    print("Wrote label-normalization sensitivity to", output)


if __name__ == "__main__":
    main()
