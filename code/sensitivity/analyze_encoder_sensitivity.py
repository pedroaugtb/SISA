#!/usr/bin/env python3
"""Compare primary MPNet outputs with an alternative embedding encoder."""

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
DIMENSIONS = (
    "Race",
    "Ethnicity",
    "Gender",
    "Disability",
    "Sexual Orientation",
    "Gender Identity",
)
SEED = 42
BLUE = "#3B6FB6"
ORANGE = "#D97706"


def args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, required=True)
    parser.add_argument("--alternative-root", type=Path, required=True)
    parser.add_argument("--output-dir", type=Path, required=True)
    parser.add_argument("--model-name", required=True)
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    return parser.parse_args()


def stable_seed(label: str) -> int:
    return (SEED + zlib.crc32(label.encode("utf-8"))) % (2**32 - 1)


def read_location_results(root: Path, kind: str, filename: str) -> pd.DataFrame:
    frames = []
    prefix = {
        "semantic": "semantic_embedding_analysis",
        "evidence": "evidence_synthesis_analysis",
    }[kind]
    suffix = "_v2" if kind == "evidence" else ""
    for location in LOCATIONS:
        path = root / f"{prefix}_{location}{suffix}" / filename
        if not path.exists():
            raise FileNotFoundError(path)
        frame = pd.read_csv(path)
        frame["location"] = location
        frames.append(frame)
    return pd.concat(frames, ignore_index=True)


def eligible_evidence(frame: pd.DataFrame) -> pd.Series:
    return (
        pd.to_numeric(frame["fetch_coverage"], errors="coerce").ge(0.50)
        & pd.to_numeric(frame["n_available_sources"], errors="coerce").ge(3)
        & pd.to_numeric(frame["evidence_alignment_source_balanced"], errors="coerce").notna()
    )


def contrast_rows(frame: pd.DataFrame, metric: str, evidence: bool) -> pd.DataFrame:
    data = frame.copy()
    if evidence:
        data = data.loc[eligible_evidence(data)].copy()
    data[metric] = pd.to_numeric(data[metric], errors="coerce")
    data = data.loc[data["condition"].isin(["minority", "majority"])]
    keys = ["location", "dimension", "domain", "outcome", "outcome_id"]
    pivot = data.pivot_table(index=keys, columns="condition", values=metric, aggfunc="mean")
    pivot = pivot.dropna(subset=["minority", "majority"]).reset_index()
    pivot["difference"] = pivot["minority"] - pivot["majority"]
    return pivot


def pooled_contrast_rows(frame: pd.DataFrame, metric: str, evidence: bool) -> pd.DataFrame:
    # Form complete focal-comparison pairs within location first. This prevents
    # a focal observation eligible only in Dallas from being paired with a
    # comparison observation eligible only in New York.
    location_pairs = contrast_rows(frame, metric, evidence)
    keys = ["dimension", "domain", "outcome", "outcome_id"]
    return location_pairs.groupby(keys, as_index=False).agg(
        minority=("minority", "mean"),
        majority=("majority", "mean"),
        difference=("difference", "mean"),
        n_locations=("location", "nunique"),
    )


def bootstrap_ci(values: np.ndarray, label: str, n_bootstrap: int) -> tuple[float, float]:
    values = values[np.isfinite(values)]
    if len(values) < 2:
        return np.nan, np.nan
    rng = np.random.default_rng(stable_seed(label))
    means = np.empty(n_bootstrap, dtype=float)
    for start in range(0, n_bootstrap, 1000):
        n = min(1000, n_bootstrap - start)
        sample = rng.choice(values, size=(n, len(values)), replace=True)
        means[start : start + n] = sample.mean(axis=1)
    low, high = np.percentile(means, [2.5, 97.5])
    return float(low), float(high)


def effect_table(
    rows: pd.DataFrame, encoder: str, metric_name: str, n_bootstrap: int, pooled: bool
) -> pd.DataFrame:
    groups = ["dimension"] if pooled else ["location", "dimension"]
    output = []
    grouper = groups[0] if len(groups) == 1 else groups
    for key, group in rows.groupby(grouper, sort=False):
        key = (key,) if not isinstance(key, tuple) else key
        record = dict(zip(groups, key))
        values = group["difference"].to_numpy(float)
        label = "::".join([metric_name, encoder, "pooled" if pooled else "location", *map(str, key)])
        low, high = bootstrap_ci(values, label, n_bootstrap)
        record.update(
            {
                "encoder": encoder,
                "metric": metric_name,
                "n_outcomes": int(len(values)),
                "mean_difference": float(np.mean(values)),
                "median_difference": float(np.median(values)),
                "ci_low": low,
                "ci_high": high,
            }
        )
        output.append(record)
    return pd.DataFrame(output)


def sign(value: float, tolerance: float = 1e-12) -> int:
    if value > tolerance:
        return 1
    if value < -tolerance:
        return -1
    return 0


def agreement_rows(primary: pd.DataFrame, alternative: pd.DataFrame, unit: str, metric: str) -> dict:
    keys = ["dimension"] if unit == "pooled_dimension" else ["location", "dimension"]
    merged = primary.merge(alternative, on=keys, suffixes=("_primary", "_alternative"))
    x = merged["mean_difference_primary"].to_numpy(float)
    y = merged["mean_difference_alternative"].to_numpy(float)
    rho, p = spearmanr(x, y) if len(merged) >= 3 else (np.nan, np.nan)
    same = np.array([sign(a) == sign(b) for a, b in zip(x, y)])
    nonzero = np.array([(sign(a) != 0 and sign(b) != 0) for a, b in zip(x, y)])
    return {
        "metric": metric,
        "unit": unit,
        "n_units": int(len(merged)),
        "same_direction_n": int(same.sum()),
        "same_direction_rate": float(same.mean()) if len(same) else np.nan,
        "same_nonzero_direction_n": int((same & nonzero).sum()),
        "spearman_rho": float(rho),
        "spearman_p_descriptive": float(p),
        "mean_absolute_effect_change": float(np.mean(np.abs(y - x))),
        "median_absolute_effect_change": float(np.median(np.abs(y - x))),
    }


def outcome_agreement(primary: pd.DataFrame, alternative: pd.DataFrame, metric: str) -> tuple[pd.DataFrame, dict]:
    keys = ["dimension", "domain", "outcome", "outcome_id"]
    merged = primary[keys + ["difference"]].merge(
        alternative[keys + ["difference"]], on=keys, suffixes=("_primary", "_alternative")
    )
    x = merged["difference_primary"].to_numpy(float)
    y = merged["difference_alternative"].to_numpy(float)
    rho, p = spearmanr(x, y) if len(merged) >= 3 else (np.nan, np.nan)
    merged["same_direction"] = [sign(a) == sign(b) for a, b in zip(x, y)]
    summary = {
        "metric": metric,
        "unit": "dimension_outcome",
        "n_units": int(len(merged)),
        "same_direction_n": int(merged["same_direction"].sum()),
        "same_direction_rate": float(merged["same_direction"].mean()),
        "same_nonzero_direction_n": int(
            sum(sign(a) == sign(b) and sign(a) != 0 for a, b in zip(x, y))
        ),
        "spearman_rho": float(rho),
        "spearman_p_descriptive": float(p),
        "mean_absolute_effect_change": float(np.mean(np.abs(y - x))),
        "median_absolute_effect_change": float(np.median(np.abs(y - x))),
    }
    return merged, summary


def pooled_mechanism(root: Path, evidence_only: bool) -> pd.DataFrame:
    frames = []
    for location in LOCATIONS:
        path = root / f"evidence_synthesis_analysis_{location}_v2" / "mechanism_map_v2.csv"
        frame = pd.read_csv(path)
        frame["location"] = location
        if evidence_only:
            frame = frame.loc[eligible_evidence(frame)].copy()
        frames.append(frame)
    data = pd.concat(frames, ignore_index=True)
    keys = ["group", "condition", "dimension", "domain", "outcome", "outcome_id"]
    numeric = [
        "url_jaccard_distance",
        "dense_distance_subject_normalized",
        "evidence_alignment_source_balanced",
        "fetch_coverage",
        "n_available_sources",
    ]
    return data.groupby(keys, as_index=False)[numeric].mean()


def block_components(frame: pd.DataFrame, x: str, y: str) -> list[tuple[str, float, float, float, int]]:
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
        parts.append((str(outcome), float(xr @ yr), float(xr @ xr), float(yr @ yr), len(xr)))
    return parts


def correlation_from_parts(parts: list[tuple[str, float, float, float, int]]) -> float:
    num = sum(p[1] for p in parts)
    x2 = sum(p[2] for p in parts)
    y2 = sum(p[3] for p in parts)
    return float(num / math.sqrt(x2 * y2))


def blocked_correlation(
    frame: pd.DataFrame, x: str, y: str, label: str, n_bootstrap: int, n_permutations: int
) -> tuple[dict, np.ndarray]:
    parts = block_components(frame, x, y)
    observed = correlation_from_parts(parts)
    nums = np.array([p[1] for p in parts]); x2 = np.array([p[2] for p in parts]); y2 = np.array([p[3] for p in parts])
    rng = np.random.default_rng(stable_seed(label))
    indices = rng.integers(0, len(parts), size=(n_bootstrap, len(parts)))
    boot = nums[indices].sum(axis=1) / np.sqrt(x2[indices].sum(axis=1) * y2[indices].sum(axis=1))
    low, high = np.percentile(boot[np.isfinite(boot)], [2.5, 97.5])

    # Match the primary analysis: permute the centered y ranks independently
    # within each outcome block while preserving the block structure.
    rank_parts = []
    for _, group in frame.groupby("outcome_id", sort=False):
        xv = pd.to_numeric(group[x], errors="coerce").to_numpy(float)
        yv = pd.to_numeric(group[y], errors="coerce").to_numpy(float)
        valid = np.isfinite(xv) & np.isfinite(yv)
        xv, yv = xv[valid], yv[valid]
        if len(xv) < 3:
            continue
        xr = rankdata(xv, method="average"); xr -= xr.mean()
        yr = rankdata(yv, method="average"); yr -= yr.mean()
        rank_parts.append((xr, yr))
    denominator = math.sqrt(x2.sum() * y2.sum())
    extreme = 0
    completed = 0
    while completed < n_permutations:
        batch = min(1000, n_permutations - completed)
        permuted_numerator = np.zeros(batch, dtype=float)
        for xr, yr in rank_parts:
            order = np.argsort(rng.random((batch, len(yr))), axis=1)
            permuted_numerator += yr[order] @ xr
        permuted = permuted_numerator / denominator
        extreme += int(np.sum(np.abs(permuted) >= abs(observed) - 1e-15))
        completed += batch
    p_value = float((extreme + 1) / (n_permutations + 1))
    row = {
        "relationship": label,
        "n_observations": int(sum(p[4] for p in parts)),
        "n_outcomes": int(len(parts)),
        "blocked_rank_correlation": observed,
        "ci_low": float(low),
        "ci_high": float(high),
        "p_blocked_permutation": p_value,
    }
    return row, boot


def correlation_difference(
    primary: pd.DataFrame,
    alternative: pd.DataFrame,
    x_primary: str,
    y_primary: str,
    x_alternative: str,
    y_alternative: str,
    label: str,
    n_bootstrap: int,
) -> dict:
    keys = ["group", "condition", "dimension", "domain", "outcome", "outcome_id"]
    left = primary[keys + [x_primary, y_primary]].rename(
        columns={x_primary: "x_primary", y_primary: "y_primary"}
    )
    right = alternative[keys + [x_alternative, y_alternative]].rename(
        columns={x_alternative: "x_alternative", y_alternative: "y_alternative"}
    )
    merged = left.merge(right, on=keys, how="inner").dropna()
    pparts = {p[0]: p for p in block_components(merged, "x_primary", "y_primary")}
    aparts = {p[0]: p for p in block_components(merged, "x_alternative", "y_alternative")}
    outcomes = sorted(set(pparts) & set(aparts))
    p = [pparts[o] for o in outcomes]; a = [aparts[o] for o in outcomes]
    p_obs = correlation_from_parts(p); a_obs = correlation_from_parts(a)
    rng = np.random.default_rng(stable_seed("difference::" + label))
    differences = np.empty(n_bootstrap)
    for i in range(n_bootstrap):
        idx = rng.integers(0, len(outcomes), size=len(outcomes))
        differences[i] = correlation_from_parts([a[j] for j in idx]) - correlation_from_parts([p[j] for j in idx])
    low, high = np.percentile(differences[np.isfinite(differences)], [2.5, 97.5])
    return {
        "relationship": label,
        "n_common_observations": int(len(merged)),
        "n_common_outcomes": int(len(outcomes)),
        "primary_r": p_obs,
        "alternative_r": a_obs,
        "difference_alternative_minus_primary": a_obs - p_obs,
        "difference_ci_low": float(low),
        "difference_ci_high": float(high),
    }


def plot_effect_agreement(effects: pd.DataFrame, output: Path) -> None:
    metrics = ["answer_displacement", "evidence_alignment"]
    titles = ["Answer semantic displacement", "Evidence alignment"]
    fig, axes = plt.subplots(1, 2, figsize=(7.2, 3.35))
    for ax, metric, title in zip(axes, metrics, titles):
        data = effects.loc[(effects["metric"] == metric) & effects["location"].isna()]
        primary = data.loc[data["encoder"] == "primary", ["dimension", "mean_difference"]]
        alternate = data.loc[data["encoder"] == "alternative", ["dimension", "mean_difference"]]
        merged = primary.merge(alternate, on="dimension", suffixes=("_primary", "_alternative"))
        ax.scatter(merged["mean_difference_primary"], merged["mean_difference_alternative"], s=42, color=ORANGE, edgecolor="white", linewidth=.6)
        values = np.r_[merged["mean_difference_primary"], merged["mean_difference_alternative"]]
        lo, hi = float(values.min()), float(values.max()); pad = max((hi - lo) * .12, .005)
        lo, hi = lo - pad, hi + pad
        ax.plot([lo, hi], [lo, hi], color="black", linewidth=.8, linestyle="--")
        ax.axhline(0, color="#777777", linewidth=.6); ax.axvline(0, color="#777777", linewidth=.6)
        for _, row in merged.iterrows():
            ax.annotate(row["dimension"], (row["mean_difference_primary"], row["mean_difference_alternative"]), xytext=(3, 3), textcoords="offset points", fontsize=6.5)
        ax.set_xlim(lo, hi); ax.set_ylim(lo, hi); ax.set_title(title, fontsize=10, weight="bold")
        ax.set_xlabel("Primary encoder effect", fontsize=9); ax.grid(False)
    axes[0].set_ylabel("Alternative encoder effect", fontsize=9)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def plot_correlations(table: pd.DataFrame, output: Path) -> None:
    labels = {
        "source_to_answer": "Source displacement × answer displacement",
        "answer_to_alignment": "Answer displacement × evidence alignment",
    }
    fig, ax = plt.subplots(figsize=(6.6, 2.7))
    y_positions = {"source_to_answer": 1, "answer_to_alignment": 0}
    offsets = {"primary": .10, "alternative": -.10}
    colors = {"primary": BLUE, "alternative": ORANGE}
    for _, row in table.iterrows():
        y = y_positions[row["relationship"]] + offsets[row["encoder"]]
        ax.errorbar(
            row["blocked_rank_correlation"], y,
            xerr=[[row["blocked_rank_correlation"] - row["ci_low"]], [row["ci_high"] - row["blocked_rank_correlation"]]],
            fmt="o", color=colors[row["encoder"]], markersize=5, capsize=2.5, linewidth=1.1,
        )
    ax.axvline(0, color="black", linewidth=.8)
    ax.set_yticks([0, 1], [labels["answer_to_alignment"], labels["source_to_answer"]])
    ax.set_xlabel("Outcome-blocked rank correlation", fontsize=9)
    ax.grid(axis="x", color="white", linewidth=.6); ax.grid(axis="y", visible=False)
    handles = [plt.Line2D([], [], marker="o", linestyle="", color=colors[k], label=k.title()) for k in ("primary", "alternative")]
    ax.legend(handles=handles, frameon=False, ncol=2, loc="lower right", fontsize=8)
    fig.tight_layout()
    fig.savefig(output.with_suffix(".png"), dpi=300, bbox_inches="tight")
    fig.savefig(output.with_suffix(".pdf"), bbox_inches="tight")
    plt.close(fig)


def main() -> None:
    options = args()
    base = options.base_dir.resolve()
    primary_root = base / "results"
    alternative_root = options.alternative_root.resolve()
    output = options.output_dir.resolve(); output.mkdir(parents=True, exist_ok=True)

    primary_sem = read_location_results(primary_root, "semantic", "group_vs_generic_semantic_metrics.csv")
    alt_sem = read_location_results(alternative_root, "semantic", "group_vs_generic_semantic_metrics.csv")
    primary_ev = read_location_results(primary_root, "evidence", "query_evidence_alignment_metrics_v2.csv")
    alt_ev = read_location_results(alternative_root, "evidence", "query_evidence_alignment_metrics_v2.csv")

    specifications = [
        ("answer_displacement", "dense_distance_subject_normalized", False, primary_sem, alt_sem),
        ("evidence_alignment", "evidence_alignment_source_balanced", True, primary_ev, alt_ev),
    ]
    effect_frames = []; agreement = []; outcome_tables = []
    pooled_cache = {}
    for metric_name, column, evidence, primary, alternate in specifications:
        primary_location = contrast_rows(primary, column, evidence)
        alt_location = contrast_rows(alternate, column, evidence)
        primary_pooled = pooled_contrast_rows(primary, column, evidence)
        alt_pooled = pooled_contrast_rows(alternate, column, evidence)
        pooled_cache[(metric_name, "primary")] = primary_pooled
        pooled_cache[(metric_name, "alternative")] = alt_pooled

        p_loc_effect = effect_table(primary_location, "primary", metric_name, options.n_bootstrap, False)
        a_loc_effect = effect_table(alt_location, "alternative", metric_name, options.n_bootstrap, False)
        p_pool_effect = effect_table(primary_pooled, "primary", metric_name, options.n_bootstrap, True)
        a_pool_effect = effect_table(alt_pooled, "alternative", metric_name, options.n_bootstrap, True)
        p_pool_effect["location"] = np.nan; a_pool_effect["location"] = np.nan
        effect_frames.extend([p_loc_effect, a_loc_effect, p_pool_effect, a_pool_effect])
        agreement.append(agreement_rows(p_pool_effect, a_pool_effect, "pooled_dimension", metric_name))
        agreement.append(agreement_rows(p_loc_effect, a_loc_effect, "location_dimension", metric_name))
        outcome_table, outcome_summary = outcome_agreement(primary_pooled, alt_pooled, metric_name)
        outcome_table.insert(0, "metric", metric_name); outcome_tables.append(outcome_table)
        agreement.append(outcome_summary)

    effects = pd.concat(effect_frames, ignore_index=True)
    effects.to_csv(output / "effects_by_encoder.csv", index=False)
    pd.DataFrame(agreement).to_csv(output / "effect_size_agreement.csv", index=False)
    pd.concat(outcome_tables, ignore_index=True).to_csv(output / "outcome_level_effect_agreement.csv", index=False)

    primary_all = pooled_mechanism(primary_root, False)
    alternate_all = pooled_mechanism(alternative_root, False)
    primary_evidence = pooled_mechanism(primary_root, True)
    alternate_evidence = pooled_mechanism(alternative_root, True)
    correlation_rows = []; boots = {}
    correlation_specs = [
        ("source_to_answer", primary_all, alternate_all, "url_jaccard_distance", "dense_distance_subject_normalized"),
        ("answer_to_alignment", primary_evidence, alternate_evidence, "dense_distance_subject_normalized", "evidence_alignment_source_balanced"),
    ]
    differences = []
    for relationship, primary, alternate, x, y in correlation_specs:
        for encoder, frame in (("primary", primary), ("alternative", alternate)):
            row, boot = blocked_correlation(frame, x, y, relationship, options.n_bootstrap, options.n_permutations)
            row["encoder"] = encoder; correlation_rows.append(row); boots[(relationship, encoder)] = boot
        differences.append(
            correlation_difference(primary, alternate, x, y, x, y, relationship, options.n_bootstrap)
        )
    correlations = pd.DataFrame(correlation_rows)
    correlations.to_csv(output / "blocked_correlations_by_encoder.csv", index=False)
    pd.DataFrame(differences).to_csv(output / "blocked_correlation_differences.csv", index=False)

    plot_effect_agreement(effects, output / "figure_encoder_effect_agreement")
    plot_correlations(correlations, output / "figure_encoder_blocked_correlations")

    summary = {
        "alternative_model": options.model_name,
        "primary_answer_encoder": "sentence-transformers/all-mpnet-base-v2",
        "primary_evidence_encoder": "sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
        "multilingual_responses_excluded": False,
        "n_bootstrap": options.n_bootstrap,
        "n_permutations": options.n_permutations,
        "effect_size_agreement": pd.DataFrame(agreement).to_dict("records"),
        "blocked_correlations": correlations.to_dict("records"),
        "blocked_correlation_differences": differences,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )
    (output / "README.md").write_text(
        "# Encoder sensitivity\n\n"
        f"Alternative encoder: `{options.model_name}`.\n\n"
        "All 819 primary responses are retained, including the 12 responses containing Portuguese. "
        "Label normalization, source recovery eligibility, source weighting, chunking parameters, "
        "outcome pooling, and inferential units follow the primary analysis. Absolute cosine levels "
        "are not compared across encoders; the analysis compares contrasts, ranks, signs, and "
        "outcome-blocked associations.\n",
        encoding="utf-8",
    )
    print("Wrote encoder comparison to", output)


if __name__ == "__main__":
    main()
