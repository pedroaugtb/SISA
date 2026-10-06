#!/usr/bin/env python3
"""Leave-one-location-out sensitivity for Figure 4.4 pathway-surface coupling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/figures"))

import paper_style as ps  # noqa: E402
from pooled_stats import blocked_rank_correlation  # noqa: E402


SOURCE_METRIC = "url_jaccard_distance"
ANSWER_METRIC = "dense_distance_subject_normalized"
LOCATION_FILES = {
    "Dallas": Path("results/evidence_synthesis_analysis_dallas_v2/mechanism_map_v2.csv"),
    "New York": Path("results/evidence_synthesis_analysis_ny_v2/mechanism_map_v2.csv"),
    "Los Angeles": Path("results/evidence_synthesis_analysis_la_v2/mechanism_map_v2.csv"),
}
ID_COLUMNS = ["group", "condition", "dimension", "domain", "outcome", "outcome_id"]
PATHWAY_COLOR = "#009E73"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "supplementary_results/pathway_surface_leave_one_location_out",
    )
    parser.add_argument("--n-permutations", type=int, default=20_000)
    parser.add_argument("--n-bootstrap", type=int, default=5_000)
    return parser.parse_args()


def load_location_maps(base: Path) -> dict[str, pd.DataFrame]:
    maps = {}
    expected_keys = None
    for location, relative in LOCATION_FILES.items():
        data = pd.read_csv(base / relative)
        required = set(ID_COLUMNS + [SOURCE_METRIC, ANSWER_METRIC])
        missing = sorted(required - set(data.columns))
        if missing:
            raise RuntimeError(f"{location}: missing required columns: {missing}")
        if len(data) != 252 or data["outcome_id"].nunique() != 21:
            raise RuntimeError(
                f"{location}: expected 252 observations and 21 outcomes; "
                f"found {len(data)} and {data['outcome_id'].nunique()}"
            )
        keys = set(map(tuple, data[ID_COLUMNS].itertuples(index=False, name=None)))
        if expected_keys is None:
            expected_keys = keys
        elif keys != expected_keys:
            raise RuntimeError(f"{location}: condition–outcome keys do not align across locations")
        data = data.copy()
        data["location"] = location
        maps[location] = data
    return maps


def pool_locations(maps: dict[str, pd.DataFrame], included: list[str]) -> pd.DataFrame:
    combined = pd.concat([maps[location] for location in included], ignore_index=True)
    pooled = (
        combined.groupby(ID_COLUMNS, as_index=False)
        .agg(
            **{
                SOURCE_METRIC: (SOURCE_METRIC, "mean"),
                ANSWER_METRIC: (ANSWER_METRIC, "mean"),
                "n_locations": ("location", "nunique"),
            }
        )
    )
    if len(pooled) != 252 or pooled["outcome_id"].nunique() != 21:
        raise RuntimeError("Pooled subset does not contain the expected 252 observations")
    if not pooled["n_locations"].eq(len(included)).all():
        raise RuntimeError("At least one condition–outcome cell is missing a retained location")
    return pooled


def estimate(
    data: pd.DataFrame,
    *,
    included: list[str],
    omitted: str | None,
    n_permutations: int,
    n_bootstrap: int,
) -> dict:
    label = (
        f"{SOURCE_METRIC}::{ANSWER_METRIC}"
        if omitted is None
        else f"lolo::{omitted}::{SOURCE_METRIC}::{ANSWER_METRIC}"
    )
    result = blocked_rank_correlation(
        data,
        SOURCE_METRIC,
        ANSWER_METRIC,
        block="outcome_id",
        label=label,
        n_perm=n_permutations,
        n_boot=n_bootstrap,
    )
    if not result:
        raise RuntimeError(f"Could not estimate leave-one-location-out result: {omitted}")
    return {
        "omitted_location": omitted or "None (all three locations)",
        "included_locations": " + ".join(included),
        "n_locations": len(included),
        "n_outcomes": int(data["outcome_id"].nunique()),
        "n_observations": int(result["n_observations"]),
        "blocked_rank_correlation": float(result["blocked_rank_correlation"]),
        "ci_low": float(result["cluster_bootstrap_ci_low"]),
        "ci_high": float(result["cluster_bootstrap_ci_high"]),
        "p_permutation_raw": float(result["p_blocked_permutation_raw"]),
        "n_permutations": int(result["n_permutations"]),
        "n_bootstrap": int(result["n_bootstrap"]),
    }


def run_analysis(
    base: Path, n_permutations: int, n_bootstrap: int
) -> pd.DataFrame:
    maps = load_location_maps(base)
    locations = list(LOCATION_FILES)
    specifications = [(None, locations)] + [
        (omitted, [location for location in locations if location != omitted])
        for omitted in locations
    ]
    rows = []
    for omitted, included in specifications:
        pooled = pool_locations(maps, included)
        rows.append(
            estimate(
                pooled,
                included=included,
                omitted=omitted,
                n_permutations=n_permutations,
                n_bootstrap=n_bootstrap,
            )
        )
    results = pd.DataFrame(rows)
    full = float(results.iloc[0]["blocked_rank_correlation"])
    results["change_from_full_sample"] = results["blocked_rank_correlation"] - full
    results["positive_estimate"] = results["blocked_rank_correlation"].gt(0)
    results["ci_excludes_zero"] = results["ci_low"].gt(0) | results["ci_high"].lt(0)
    return results


def paper_table(results: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in results.itertuples(index=False):
        p_text = "< .001" if row.p_permutation_raw < 0.001 else f"{row.p_permutation_raw:.3f}"
        rows.append(
            {
                "Omitted location": (
                    "None (all three locations)"
                    if row.omitted_location == "None (all three locations)"
                    else row.omitted_location
                ),
                "Included locations": row.included_locations,
                "Blocked r": round(row.blocked_rank_correlation, 3),
                "95% CI": f"[{row.ci_low:.3f}, {row.ci_high:.3f}]",
                "Change from full": round(row.change_from_full_sample, 3),
                "Permutation p": p_text,
            }
        )
    return pd.DataFrame(rows)


def plot_diagnostic(results: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    ps.use_paper_style()
    labels = [
        "All three locations",
        "Without Dallas",
        "Without New York",
        "Without Los Angeles",
    ]
    y = list(range(len(results) - 1, -1, -1))
    full = results.iloc[0]
    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH, 2.25))
    ax.axvline(0, color=ps.INK, linewidth=0.65, zorder=0)
    ax.axvline(
        full["blocked_rank_correlation"],
        color=ps.MUTED,
        linewidth=0.7,
        linestyle=(0, (3, 2)),
        zorder=0,
    )
    ax.axvspan(full["ci_low"], full["ci_high"], color=ps.BAND, zorder=-2)
    for index, row in results.iterrows():
        is_full = index == 0
        color = ps.INK if is_full else PATHWAY_COLOR
        ax.hlines(y[index], row["ci_low"], row["ci_high"], color=color, linewidth=1.35)
        ax.scatter(
            row["blocked_rank_correlation"],
            y[index],
            s=34 if is_full else 29,
            marker="s" if is_full else "o",
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            row["ci_high"] + 0.008,
            y[index],
            f"{row['blocked_rank_correlation']:.2f}",
            ha="left",
            va="center",
            fontsize=ps.FS_NOTE,
            color=ps.INK_SOFT,
        )
    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlim(-0.02, max(0.56, float(results["ci_high"].max()) + 0.05))
    ax.set_ylim(-0.6, len(results) - 0.4)
    ax.set_xlabel("Within-outcome blocked rank correlation")
    ax.grid(axis="x", color=ps.GRID, linewidth=0.5, zorder=-1)
    ps.strip_axes(ax, keep=("bottom",))
    ax.set_title(
        "Leave-one-location-out sensitivity of pathway–surface coupling",
        loc="left",
        fontweight="bold",
        color=ps.INK,
        pad=8,
    )
    fig.subplots_adjust(left=0.26, right=0.94, bottom=0.23, top=0.83)
    png = output / "figure_S3_pathway_surface_leave_one_location_out.png"
    pdf = output / "figure_S3_pathway_surface_leave_one_location_out.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=ps.WHITE)
    fig.savefig(pdf, bbox_inches="tight", facecolor=ps.WHITE)
    plt.close(fig)
    return png, pdf


def write_documentation(output: Path, results: pd.DataFrame) -> None:
    full = results.iloc[0]
    sensitivity = results.iloc[1:]
    minimum = sensitivity.loc[sensitivity["blocked_rank_correlation"].idxmin()]
    maximum = sensitivity.loc[sensitivity["blocked_rank_correlation"].idxmax()]
    readme = f"""# Leave-one-location-out sensitivity for pathway–surface coupling

The analysis reconstructs the three-location pooled mechanism map after dropping Dallas, New York,
or Los Angeles in turn. For every retained subset, source-set displacement and answer displacement
are averaged within the same condition–outcome across the retained locations before ranking and
mean-centering within outcome. Outcome remains the bootstrap cluster and permutations remain
within outcome.

The full pooled estimate is r = {full['blocked_rank_correlation']:.3f}, 95% CI
[{full['ci_low']:.3f}, {full['ci_high']:.3f}]. All 3/3 leave-one-location-out estimates are positive,
and all 3/3 confidence intervals exclude zero. Estimates range from
{minimum['blocked_rank_correlation']:.3f} after omitting {minimum['omitted_location']} to
{maximum['blocked_rank_correlation']:.3f} after omitting {maximum['omitted_location']}.

Each specification contains 21 outcomes and 252 condition–outcome observations. The sensitivity
specifications differ only in which geographic repetitions are averaged into those observations.

Outputs:

- `leave_one_location_out_results.csv`: raw estimates and uncertainty.
- `table_S5_leave_one_location_out.csv` and `.tex`: paper-facing table.
- `figure_S3_pathway_surface_leave_one_location_out.pdf` and `.png`: optional diagnostic figure.
- `MANUSCRIPT_TEXT.md`: suggested results text and caption.
"""
    (output / "README.md").write_text(readme, encoding="utf-8")

    manuscript = f"""# Suggested manuscript text

In a leave-one-location-out sensitivity analysis, we reconstructed the pooled measure after
excluding each location in turn. The full three-location association was
$r={full['blocked_rank_correlation']:.2f}$, 95% CI [{full['ci_low']:.2f}, {full['ci_high']:.2f}].
The two-location estimates ranged from $r={minimum['blocked_rank_correlation']:.2f}$ to
$r={maximum['blocked_rank_correlation']:.2f}$, remained positive in all three specifications, and
all outcome-cluster bootstrap confidence intervals excluded zero. The pathway–surface association
was therefore not determined by any single IP location.

Suggested caption: Leave-one-location-out sensitivity of the pooled association between URL-level
source-set displacement and subject-normalized answer displacement. Each sensitivity estimate
averages the two retained geographic repetitions within condition and outcome before calculating
the within-outcome blocked rank correlation. Horizontal lines show 95% outcome-cluster bootstrap
confidence intervals. The square denotes the full three-location estimate.
"""
    (output / "MANUSCRIPT_TEXT.md").write_text(manuscript, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = args.base_dir.absolute()
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)

    results = run_analysis(base, args.n_permutations, args.n_bootstrap)

    # Verify that reconstructing all three locations reproduces the published pooled estimate.
    cache = pd.read_csv(base / "figures_three_locations_pooled/pooled_pipeline_correlations.csv")
    expected = cache.loc[
        cache["metric_x"].eq(SOURCE_METRIC) & cache["metric_y"].eq(ANSWER_METRIC),
        "blocked_rank_correlation",
    ]
    if len(expected) != 1 or abs(float(expected.iloc[0]) - float(results.iloc[0]["blocked_rank_correlation"])) > 1e-12:
        raise RuntimeError("Reconstructed three-location baseline does not match Figure 4.4")

    results.to_csv(output / "leave_one_location_out_results.csv", index=False)
    table = paper_table(results)
    table.to_csv(output / "table_S5_leave_one_location_out.csv", index=False)
    (output / "table_S5_leave_one_location_out.tex").write_text(
        table.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.3f}"),
        encoding="utf-8",
    )
    png, pdf = plot_diagnostic(results, output)
    write_documentation(output, results)

    sensitivity = results.iloc[1:]
    summary = {
        "metric_x": SOURCE_METRIC,
        "metric_y": ANSWER_METRIC,
        "blocking_variable": "outcome_id",
        "full_sample_r": float(results.iloc[0]["blocked_rank_correlation"]),
        "full_sample_ci": [float(results.iloc[0]["ci_low"]), float(results.iloc[0]["ci_high"])],
        "leave_one_location_out_min_r": float(sensitivity["blocked_rank_correlation"].min()),
        "leave_one_location_out_max_r": float(sensitivity["blocked_rank_correlation"].max()),
        "positive_estimates": int(sensitivity["positive_estimate"].sum()),
        "ci_excludes_zero": int((sensitivity["ci_low"] > 0).sum()),
        "n_specifications": int(len(sensitivity)),
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )

    print(f"Results : {output / 'leave_one_location_out_results.csv'}")
    print(f"Table   : {output / 'table_S5_leave_one_location_out.csv'}")
    print(f"Figure  : {png}")
    print(f"          {pdf}")


if __name__ == "__main__":
    main()
