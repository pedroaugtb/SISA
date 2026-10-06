#!/usr/bin/env python3
"""Compare primary, smaller, and high-context evidence chunking configurations."""

from __future__ import annotations

import argparse
import json
import math
import zlib
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.stats import rankdata, spearmanr


LOCATIONS = ("dallas", "ny", "la")
CONFIGURATIONS = ("primary", "smaller", "high_context")
ALTERNATIVES = CONFIGURATIONS[1:]
METRIC = "evidence_alignment_source_balanced"
ANSWER = "dense_distance_subject_normalized"
SEED = 42
COLORS = {"primary": "#3B6FB6", "smaller": "#D97706", "high_context": "#2E8B57"}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--sensitivity-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    return parser.parse_args()


def stable_seed(label: str) -> int:
    return (SEED + zlib.crc32(label.encode("utf-8"))) % (2**32 - 1)


def result_root(base: Path, sensitivity_root: Path, configuration: str) -> Path:
    if configuration == "primary":
        return base / "results"
    return sensitivity_root / "raw" / configuration


def read_mechanism(root: Path) -> pd.DataFrame:
    frames = []
    for location in LOCATIONS:
        path = root / f"evidence_synthesis_analysis_{location}_v2" / "mechanism_map_v2.csv"
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["location"] = location
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    if data.duplicated(["location", "file"]).any():
        raise ValueError("Expected location + file to identify one condition row.")
    return data


def primary_eligible_keys(primary: pd.DataFrame) -> pd.MultiIndex:
    eligible = (
        pd.to_numeric(primary["fetch_coverage"], errors="coerce").ge(0.50)
        & pd.to_numeric(primary["n_available_sources"], errors="coerce").ge(3)
        & pd.to_numeric(primary[METRIC], errors="coerce").notna()
    )
    return pd.MultiIndex.from_frame(primary.loc[eligible, ["location", "file"]])


def freeze_sample(frame: pd.DataFrame, keys: pd.MultiIndex, configuration: str) -> pd.DataFrame:
    indexed = frame.set_index(["location", "file"], drop=False)
    missing = keys.difference(indexed.index)
    if len(missing):
        raise ValueError(f"{configuration} is missing {len(missing)} primary-eligible rows.")
    frozen = indexed.loc[keys].reset_index(drop=True)
    required = [METRIC, ANSWER]
    invalid = frozen[required].apply(pd.to_numeric, errors="coerce").isna().any(axis=1)
    if invalid.any():
        raise ValueError(
            f"{configuration} has {int(invalid.sum())} invalid primary-eligible rows."
        )
    return frozen


def location_pairs(frame: pd.DataFrame) -> pd.DataFrame:
    data = frame.loc[frame["condition"].isin(["minority", "majority"])].copy()
    data[METRIC] = pd.to_numeric(data[METRIC], errors="coerce")
    keys = ["location", "dimension", "domain", "outcome", "outcome_id"]
    pivot = data.pivot_table(index=keys, columns="condition", values=METRIC, aggfunc="mean")
    pivot = pivot.dropna(subset=["minority", "majority"]).reset_index()
    pivot["difference"] = pivot["minority"] - pivot["majority"]
    return pivot


def pooled_outcome_pairs(location_effects: pd.DataFrame) -> pd.DataFrame:
    keys = ["dimension", "domain", "outcome", "outcome_id"]
    return location_effects.groupby(keys, as_index=False).agg(
        minority=("minority", "mean"),
        majority=("majority", "mean"),
        difference=("difference", "mean"),
        n_locations=("location", "nunique"),
    )


def bootstrap_mean_ci(values: np.ndarray, label: str, n_bootstrap: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
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
    if len(values) <= 18:
        total = 2 ** len(values)
        extreme = 0
        powers = 1 << np.arange(len(values), dtype=np.uint64)
        for start in range(0, total, 10000):
            stop = min(total, start + 10000)
            integers = np.arange(start, stop, dtype=np.uint64)[:, None]
            signs = ((integers & powers) > 0).astype(float) * 2.0 - 1.0
            extreme += int(np.sum(np.abs((signs * values).mean(axis=1)) >= observed - 1e-15))
        return float(extreme / total)
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
    pooled: pd.DataFrame, configuration: str, n_bootstrap: int, n_permutations: int
) -> pd.DataFrame:
    rows = []
    for dimension, group in pooled.groupby("dimension", sort=True):
        values = group["difference"].to_numpy(float)
        low, high = bootstrap_mean_ci(values, f"{configuration}::{dimension}", n_bootstrap)
        rows.append(
            {
                "configuration": configuration,
                "dimension": dimension,
                "n_outcomes": int(len(values)),
                "mean_alignment_difference": float(values.mean()),
                "median_alignment_difference": float(np.median(values)),
                "bootstrap_ci_low": low,
                "bootstrap_ci_high": high,
                "focal_mean_alignment": float(group["minority"].mean()),
                "comparison_mean_alignment": float(group["majority"].mean()),
                "mean_locations_per_outcome": float(group["n_locations"].mean()),
                "p_signflip_raw": signflip_p(
                    values, f"{configuration}::{dimension}", n_permutations
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
    primary: pd.DataFrame,
    alternative: pd.DataFrame,
    keys: list[str],
    value: str,
    unit: str,
    alternative_name: str,
) -> tuple[pd.DataFrame, dict]:
    left = primary[keys + [value]].rename(columns={value: "effect_primary"})
    right = alternative[keys + [value]].rename(columns={value: "effect_alternative"})
    merged = left.merge(right, on=keys, how="inner", validate="one_to_one")
    x = merged["effect_primary"].to_numpy(float)
    y = merged["effect_alternative"].to_numpy(float)
    rho, p_value = spearmanr(x, y) if len(merged) >= 3 else (np.nan, np.nan)
    merged["same_direction"] = [direction(a) == direction(b) for a, b in zip(x, y)]
    summary = {
        "alternative": alternative_name,
        "unit": unit,
        "n_units": int(len(merged)),
        "same_direction_n": int(merged["same_direction"].sum()),
        "same_direction_rate": float(merged["same_direction"].mean()),
        "spearman_rho": float(rho),
        "spearman_p_descriptive": float(p_value),
        "mean_absolute_effect_change": float(np.mean(np.abs(y - x))),
        "median_absolute_effect_change": float(np.median(np.abs(y - x))),
    }
    merged.insert(0, "alternative", alternative_name)
    merged.insert(1, "unit", unit)
    return merged, summary


def pooled_conditions(frame: pd.DataFrame) -> pd.DataFrame:
    keys = ["group", "condition", "dimension", "domain", "outcome", "outcome_id"]
    numeric = [ANSWER, METRIC]
    return frame.groupby(keys, as_index=False).agg(
        **{column: (column, "mean") for column in numeric},
        n_locations=("location", "nunique"),
    )


def block_parts(frame: pd.DataFrame, x: str, y: str) -> list[tuple[str, np.ndarray, np.ndarray]]:
    parts = []
    for outcome, group in frame.groupby("outcome_id", sort=False):
        xv = pd.to_numeric(group[x], errors="coerce").to_numpy(float)
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
    frame: pd.DataFrame, configuration: str, n_bootstrap: int, n_permutations: int
) -> tuple[dict, dict[str, tuple[str, np.ndarray, np.ndarray]]]:
    parts = block_parts(frame, ANSWER, METRIC)
    if len(parts) < 3:
        raise ValueError(f"Too few outcome blocks for {configuration}.")
    observed = correlation_from_parts(parts)
    nums = np.array([float(xr @ yr) for _, xr, yr in parts])
    x2 = np.array([float(xr @ xr) for _, xr, _ in parts])
    y2 = np.array([float(yr @ yr) for _, _, yr in parts])
    rng = np.random.default_rng(stable_seed("blocked::" + configuration))
    indices = rng.integers(0, len(parts), size=(n_bootstrap, len(parts)))
    boot = nums[indices].sum(axis=1) / np.sqrt(
        x2[indices].sum(axis=1) * y2[indices].sum(axis=1)
    )
    low, high = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5])
    denominator = math.sqrt(x2.sum() * y2.sum())
    extreme = 0
    complete = 0
    while complete < n_permutations:
        size = min(1000, n_permutations - complete)
        permuted_numerator = np.zeros(size, dtype=float)
        for _, xr, yr in parts:
            order = np.argsort(rng.random((size, len(yr))), axis=1)
            permuted_numerator += yr[order] @ xr
        permuted = permuted_numerator / denominator
        extreme += int(np.sum(np.abs(permuted) >= abs(observed) - 1e-15))
        complete += size
    row = {
        "configuration": configuration,
        "relationship": "answer_displacement_to_evidence_alignment",
        "n_observations": int(sum(len(xr) for _, xr, _ in parts)),
        "n_outcomes": int(len(parts)),
        "blocked_rank_correlation": observed,
        "cluster_bootstrap_ci_low": float(low),
        "cluster_bootstrap_ci_high": float(high),
        "p_blocked_permutation": float((extreme + 1) / (n_permutations + 1)),
    }
    return row, {part[0]: part for part in parts}


def correlation_difference(
    primary_parts: dict[str, tuple[str, np.ndarray, np.ndarray]],
    alternative_parts: dict[str, tuple[str, np.ndarray, np.ndarray]],
    alternative: str,
    n_bootstrap: int,
) -> dict:
    outcomes = sorted(set(primary_parts) & set(alternative_parts))
    primary = [primary_parts[key] for key in outcomes]
    other = [alternative_parts[key] for key in outcomes]
    primary_r = correlation_from_parts(primary)
    alternative_r = correlation_from_parts(other)
    rng = np.random.default_rng(stable_seed("correlation-difference::" + alternative))
    differences = np.empty(n_bootstrap, dtype=float)
    for index in range(n_bootstrap):
        sample = rng.integers(0, len(outcomes), size=len(outcomes))
        differences[index] = correlation_from_parts([other[i] for i in sample]) - correlation_from_parts(
            [primary[i] for i in sample]
        )
    low, high = np.percentile(differences[np.isfinite(differences)], [2.5, 97.5])
    return {
        "alternative": alternative,
        "n_common_outcomes": int(len(outcomes)),
        "primary_r": primary_r,
        "alternative_r": alternative_r,
        "difference_alternative_minus_primary": alternative_r - primary_r,
        "difference_ci_low": float(low),
        "difference_ci_high": float(high),
    }


def plot_effect_agreement(dimension_effects: pd.DataFrame, output: Path) -> None:
    primary = dimension_effects.loc[
        dimension_effects["configuration"] == "primary",
        ["dimension", "mean_alignment_difference"],
    ]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.25), sharex=True, sharey=True)
    for ax, alternative in zip(axes, ALTERNATIVES):
        other = dimension_effects.loc[
            dimension_effects["configuration"] == alternative,
            ["dimension", "mean_alignment_difference"],
        ]
        merged = primary.merge(other, on="dimension", suffixes=("_primary", "_alternative"))
        x = merged["mean_alignment_difference_primary"]
        y = merged["mean_alignment_difference_alternative"]
        values = np.r_[x, y]
        pad = max(float(values.max() - values.min()) * 0.12, 0.005)
        low, high = float(values.min() - pad), float(values.max() + pad)
        ax.scatter(x, y, s=44, color=COLORS[alternative], edgecolor="white", linewidth=0.6)
        ax.plot([low, high], [low, high], color="black", linewidth=0.8, linestyle="--")
        ax.axhline(0, color="#777777", linewidth=0.6)
        ax.axvline(0, color="#777777", linewidth=0.6)
        for _, row in merged.iterrows():
            ax.annotate(
                row["dimension"],
                (row["mean_alignment_difference_primary"], row["mean_alignment_difference_alternative"]),
                xytext=(3, 3), textcoords="offset points", fontsize=6.3,
            )
        ax.set_xlim(low, high); ax.set_ylim(low, high)
        ax.set_title(alternative.replace("_", " ").title(), fontsize=10, weight="bold")
        ax.set_xlabel("Primary alignment effect", fontsize=9)
    axes[0].set_ylabel("Alternative alignment effect", fontsize=9)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_correlations(table: pd.DataFrame, output: Path) -> None:
    fig, ax = plt.subplots(figsize=(6.0, 2.7))
    positions = {name: index for index, name in enumerate(CONFIGURATIONS[::-1])}
    for _, row in table.iterrows():
        y = positions[row["configuration"]]
        ax.errorbar(
            row["blocked_rank_correlation"], y,
            xerr=[
                [row["blocked_rank_correlation"] - row["cluster_bootstrap_ci_low"]],
                [row["cluster_bootstrap_ci_high"] - row["blocked_rank_correlation"]],
            ],
            fmt="o", color=COLORS[row["configuration"]], markersize=5, capsize=2.5,
        )
    ax.axvline(0, color="black", linewidth=0.8)
    ax.set_yticks(list(positions.values()), [name.replace("_", " ").title() for name in positions])
    ax.set_xlabel("Outcome-blocked rank correlation", fontsize=9)
    ax.set_title("Answer displacement × evidence alignment", fontsize=10, weight="bold")
    ax.grid(axis="x", color="#dddddd", linewidth=0.6); ax.grid(axis="y", visible=False)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    options = parse_args()
    base = options.base_dir.resolve()
    sensitivity_root = options.sensitivity_root.resolve()
    output = options.output_dir.resolve()
    output.mkdir(parents=True, exist_ok=True)

    raw = {
        configuration: read_mechanism(result_root(base, sensitivity_root, configuration))
        for configuration in CONFIGURATIONS
    }
    eligible_keys = primary_eligible_keys(raw["primary"])
    frozen = {
        configuration: freeze_sample(frame, eligible_keys, configuration)
        for configuration, frame in raw.items()
    }

    location_cache = {name: location_pairs(frame) for name, frame in frozen.items()}
    pooled_cache = {name: pooled_outcome_pairs(frame) for name, frame in location_cache.items()}
    dimension_effects = pd.concat(
        [
            dimension_table(
                pooled_cache[name], name, options.n_bootstrap, options.n_permutations
            )
            for name in CONFIGURATIONS
        ],
        ignore_index=True,
    )
    dimension_effects.to_csv(output / "dimension_effects_by_configuration.csv", index=False)

    agreement_rows = []
    agreement_details = []
    primary_dimensions = dimension_effects.loc[dimension_effects["configuration"] == "primary"]
    location_dimension = {
        name: frame.groupby(["location", "dimension"], as_index=False)["difference"].mean()
        for name, frame in location_cache.items()
    }
    for alternative in ALTERNATIVES:
        details, summary = agreement_summary(
            primary_dimensions,
            dimension_effects.loc[dimension_effects["configuration"] == alternative],
            ["dimension"], "mean_alignment_difference", "pooled_dimension", alternative,
        )
        agreement_details.append(details); agreement_rows.append(summary)
        details, summary = agreement_summary(
            location_dimension["primary"], location_dimension[alternative],
            ["location", "dimension"], "difference", "location_dimension", alternative,
        )
        agreement_details.append(details); agreement_rows.append(summary)
        details, summary = agreement_summary(
            pooled_cache["primary"], pooled_cache[alternative],
            ["dimension", "domain", "outcome", "outcome_id"],
            "difference", "dimension_outcome", alternative,
        )
        agreement_details.append(details); agreement_rows.append(summary)
    agreement = pd.DataFrame(agreement_rows)
    agreement.to_csv(output / "effect_stability_summary.csv", index=False)
    pd.concat(agreement_details, ignore_index=True).to_csv(
        output / "effect_stability_details.csv", index=False
    )

    correlation_rows = []
    correlation_parts = {}
    for configuration in CONFIGURATIONS:
        row, parts = blocked_correlation(
            pooled_conditions(frozen[configuration]),
            configuration,
            options.n_bootstrap,
            options.n_permutations,
        )
        correlation_rows.append(row)
        correlation_parts[configuration] = parts
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output / "blocked_correlations_by_configuration.csv", index=False)
    differences = pd.DataFrame(
        [
            correlation_difference(
                correlation_parts["primary"],
                correlation_parts[alternative],
                alternative,
                options.n_bootstrap,
            )
            for alternative in ALTERNATIVES
        ]
    )
    differences.to_csv(output / "blocked_correlation_differences.csv", index=False)

    sample_audit = pd.DataFrame(
        [
            {
                "configuration": name,
                "primary_eligible_rows": int(len(frame)),
                "locations": int(frame["location"].nunique()),
                "outcomes": int(frame["outcome_id"].nunique()),
            }
            for name, frame in frozen.items()
        ]
    )
    sample_audit.to_csv(output / "sample_audit.csv", index=False)

    plot_effect_agreement(dimension_effects, output / "figure_chunking_effect_agreement")
    plot_correlations(correlations, output / "figure_chunking_blocked_correlations")

    summary = {
        "estimand": "focal minus comparison evidence alignment",
        "eligibility": "Frozen from primary (fetch coverage >= .50 and >= 3 usable sources).",
        "absolute_cosine_levels_compared": False,
        "n_bootstrap": options.n_bootstrap,
        "n_permutations": options.n_permutations,
        "effect_stability": agreement.to_dict("records"),
        "blocked_correlations": correlations.to_dict("records"),
        "blocked_correlation_differences": differences.to_dict("records"),
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Evidence-alignment chunking sensitivity\n\n"
        "This analysis compares the effective primary configuration (source 112/40; "
        "answer 96/24) with a smaller-window configuration (80/30; 64/20) and a "
        "same-encoder high-context/high-overlap configuration (112/56; 112/32).\n\n"
        "The source window cannot defensibly exceed 112 under the primary encoder: the "
        "production pipeline caps requested windows at `model.max_seq_length - 16`. "
        "Calling a nominal 160-token rerun would therefore reproduce 112-token windows.\n\n"
        "Eligibility is frozen from the primary analysis by location and response file. "
        "Effects are evidence alignment for focal minus comparison conditions. Location "
        "pairs are formed before repetitions are averaged within outcome. Absolute cosine "
        "levels are not treated as comparable across chunking configurations; conclusions "
        "use effect directions, ranks, and outcome-blocked associations.\n",
        encoding="utf-8",
    )
    print("Wrote chunking comparison to", output)


if __name__ == "__main__":
    main()
