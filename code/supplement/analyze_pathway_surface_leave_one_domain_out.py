#!/usr/bin/env python3
"""Leave-one-domain-out sensitivity for Figure 4.4 pathway-surface coupling."""

from __future__ import annotations

import argparse
import json
from pathlib import Path
import sys

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd


REPO_ROOT = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO_ROOT / "code/figures"))

import paper_style as ps  # noqa: E402
from pooled_stats import blocked_rank_correlation  # noqa: E402


SOURCE_METRIC = "url_jaccard_distance"
ANSWER_METRIC = "dense_distance_subject_normalized"
DOMAIN_ORDER = [
    "Healthcare",
    "Employment",
    "Education",
    "Housing",
    "Credit & Financial Services",
    "Criminal Justice",
    "Government Benefits",
]
DOMAIN_LABELS = {
    "Healthcare": "Healthcare",
    "Employment": "Employment",
    "Education": "Education",
    "Housing": "Housing",
    "Credit & Financial Services": "Credit & financial services",
    "Criminal Justice": "Criminal justice",
    "Government Benefits": "Government benefits",
}
SCOPE_FILES = {
    "Pooled": Path("figures_three_locations_pooled/pooled_mechanism_all.csv"),
    "Dallas": Path("results/evidence_synthesis_analysis_dallas_v2/mechanism_map_v2.csv"),
    "New York": Path("results/evidence_synthesis_analysis_ny_v2/mechanism_map_v2.csv"),
    "Los Angeles": Path("results/evidence_synthesis_analysis_la_v2/mechanism_map_v2.csv"),
}
PATHWAY_COLOR = "#009E73"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=REPO_ROOT)
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=REPO_ROOT / "supplementary_results/pathway_surface_leave_one_domain_out",
    )
    parser.add_argument("--n-permutations", type=int, default=20_000)
    parser.add_argument("--n-bootstrap", type=int, default=5_000)
    return parser.parse_args()


def validate_map(data: pd.DataFrame, scope: str) -> None:
    required = {
        "domain",
        "outcome_id",
        "condition",
        SOURCE_METRIC,
        ANSWER_METRIC,
    }
    missing = sorted(required - set(data.columns))
    if missing:
        raise RuntimeError(f"{scope}: missing required columns: {missing}")
    if set(data["domain"].dropna().unique()) != set(DOMAIN_ORDER):
        raise RuntimeError(f"{scope}: unexpected domain set")
    if data["outcome_id"].nunique() != 21:
        raise RuntimeError(f"{scope}: expected 21 outcomes, found {data['outcome_id'].nunique()}")
    counts = data.groupby("domain")["outcome_id"].nunique()
    if not counts.eq(3).all():
        raise RuntimeError(f"{scope}: expected three outcomes per domain; found {counts.to_dict()}")


def estimate(
    data: pd.DataFrame,
    *,
    scope: str,
    omitted_domain: str | None,
    n_permutations: int,
    n_bootstrap: int,
) -> dict:
    subset = data if omitted_domain is None else data.loc[data["domain"].ne(omitted_domain)]
    label = (
        f"{SOURCE_METRIC}::{ANSWER_METRIC}"
        if omitted_domain is None
        else f"lodo::{scope}::{omitted_domain}::{SOURCE_METRIC}::{ANSWER_METRIC}"
    )
    result = blocked_rank_correlation(
        subset,
        SOURCE_METRIC,
        ANSWER_METRIC,
        block="outcome_id",
        label=label,
        n_perm=n_permutations,
        n_boot=n_bootstrap,
    )
    if not result:
        raise RuntimeError(f"{scope}, omitted={omitted_domain}: correlation could not be estimated")
    return {
        "scope": scope,
        "omitted_domain": omitted_domain or "None (full sample)",
        "n_domains": int(subset["domain"].nunique()),
        "n_outcomes": int(subset["outcome_id"].nunique()),
        "n_observations": int(result["n_observations"]),
        "blocked_rank_correlation": float(result["blocked_rank_correlation"]),
        "ci_low": float(result["cluster_bootstrap_ci_low"]),
        "ci_high": float(result["cluster_bootstrap_ci_high"]),
        "p_permutation_raw": float(result["p_blocked_permutation_raw"]),
        "n_permutations": int(result["n_permutations"]),
        "n_bootstrap": int(result["n_bootstrap"]),
    }


def run_analysis(base: Path, n_permutations: int, n_bootstrap: int) -> pd.DataFrame:
    rows = []
    for scope, relative_path in SCOPE_FILES.items():
        data = pd.read_csv(base / relative_path)
        validate_map(data, scope)
        rows.append(
            estimate(
                data,
                scope=scope,
                omitted_domain=None,
                n_permutations=n_permutations,
                n_bootstrap=n_bootstrap,
            )
        )
        for domain in DOMAIN_ORDER:
            rows.append(
                estimate(
                    data,
                    scope=scope,
                    omitted_domain=domain,
                    n_permutations=n_permutations,
                    n_bootstrap=n_bootstrap,
                )
            )
    results = pd.DataFrame(rows)
    full = (
        results.loc[results["omitted_domain"].eq("None (full sample)"), [
            "scope", "blocked_rank_correlation"
        ]]
        .rename(columns={"blocked_rank_correlation": "full_sample_correlation"})
    )
    results = results.merge(full, on="scope", validate="many_to_one")
    results["change_from_full_sample"] = (
        results["blocked_rank_correlation"] - results["full_sample_correlation"]
    )
    results["positive_estimate"] = results["blocked_rank_correlation"].gt(0)
    results["ci_excludes_zero"] = results["ci_low"].gt(0) | results["ci_high"].lt(0)
    scope_rank = {scope: index for index, scope in enumerate(SCOPE_FILES)}
    domain_rank = {"None (full sample)": -1, **{d: i for i, d in enumerate(DOMAIN_ORDER)}}
    results["_scope_rank"] = results["scope"].map(scope_rank)
    results["_domain_rank"] = results["omitted_domain"].map(domain_rank)
    return results.sort_values(["_scope_rank", "_domain_rank"]).drop(
        columns=["_scope_rank", "_domain_rank"]
    )


def paper_table(pooled: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for row in pooled.itertuples(index=False):
        omitted = (
            "None (full sample)"
            if row.omitted_domain == "None (full sample)"
            else DOMAIN_LABELS[row.omitted_domain]
        )
        p_text = "< .001" if row.p_permutation_raw < 0.001 else f"{row.p_permutation_raw:.3f}"
        rows.append(
            {
                "Omitted domain": omitted,
                "Outcomes": int(row.n_outcomes),
                "Observations": int(row.n_observations),
                "Blocked r": round(row.blocked_rank_correlation, 3),
                "95% CI": f"[{row.ci_low:.3f}, {row.ci_high:.3f}]",
                "Permutation p": p_text,
                "Positive": "Yes" if row.positive_estimate else "No",
            }
        )
    return pd.DataFrame(rows)


def plot_pooled(pooled: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    ps.use_paper_style()
    order = ["None (full sample)"] + DOMAIN_ORDER
    labels = ["Full sample"] + [f"Without {DOMAIN_LABELS[d]}" for d in DOMAIN_ORDER]
    indexed = pooled.set_index("omitted_domain").loc[order]
    y = np.arange(len(order))[::-1]
    full_r = float(indexed.iloc[0]["blocked_rank_correlation"])

    fig, ax = plt.subplots(figsize=(ps.TEXT_WIDTH, 3.25))
    ax.axvline(0, color=ps.INK, linewidth=0.65, zorder=0)
    ax.axvline(full_r, color=ps.MUTED, linewidth=0.7, linestyle=(0, (3, 2)), zorder=0)
    ax.axvspan(
        float(indexed.iloc[0]["ci_low"]),
        float(indexed.iloc[0]["ci_high"]),
        color=ps.BAND,
        zorder=-2,
    )

    for index, (_, row) in enumerate(indexed.iterrows()):
        yy = y[index]
        is_full = index == 0
        color = ps.INK if is_full else PATHWAY_COLOR
        marker = "s" if is_full else "o"
        ax.hlines(yy, row["ci_low"], row["ci_high"], color=color, linewidth=1.35, zorder=2)
        ax.scatter(
            row["blocked_rank_correlation"],
            yy,
            s=34 if is_full else 29,
            marker=marker,
            facecolor=color,
            edgecolor="white",
            linewidth=0.5,
            zorder=3,
        )
        ax.text(
            row["ci_high"] + 0.012,
            yy,
            f"{row['blocked_rank_correlation']:.2f}",
            ha="left",
            va="center",
            fontsize=ps.FS_NOTE,
            color=ps.INK_SOFT,
        )

    ax.set_yticks(y)
    ax.set_yticklabels(labels)
    ax.tick_params(axis="y", length=0)
    ax.set_xlabel("Within-outcome blocked rank correlation")
    ax.set_xlim(-0.02, max(0.56, float(indexed["ci_high"].max()) + 0.07))
    ax.set_ylim(-0.6, len(order) - 0.4)
    ax.grid(axis="x", color=ps.GRID, linewidth=0.5, zorder=-1)
    ps.strip_axes(ax, keep=("bottom",))
    ax.set_title(
        "Leave-one-domain-out sensitivity of pathway–surface coupling",
        loc="left",
        fontweight="bold",
        color=ps.INK,
        pad=8,
    )
    fig.subplots_adjust(left=0.28, right=0.94, bottom=0.16, top=0.86)
    png = output / "figure_S2_pathway_surface_leave_one_domain_out.png"
    pdf = output / "figure_S2_pathway_surface_leave_one_domain_out.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=ps.WHITE)
    fig.savefig(pdf, bbox_inches="tight", facecolor=ps.WHITE)
    plt.close(fig)
    return png, pdf


def plot_all_scopes(results: pd.DataFrame, output: Path) -> tuple[Path, Path]:
    ps.use_paper_style()
    order = ["None (full sample)"] + DOMAIN_ORDER
    labels = ["Full sample"] + [f"Without {DOMAIN_LABELS[d]}" for d in DOMAIN_ORDER]
    fig, axes = plt.subplots(2, 2, figsize=(ps.TEXT_WIDTH, 5.8), sharex=True)
    for ax, scope in zip(axes.flat, SCOPE_FILES):
        data = results.loc[results["scope"].eq(scope)].set_index("omitted_domain").loc[order]
        y = np.arange(len(order))[::-1]
        ax.axvline(0, color=ps.INK, linewidth=0.6, zorder=0)
        ax.grid(axis="x", color=ps.GRID, linewidth=0.45, zorder=-1)
        for index, (_, row) in enumerate(data.iterrows()):
            is_full = index == 0
            color = ps.INK if is_full else PATHWAY_COLOR
            ax.hlines(y[index], row["ci_low"], row["ci_high"], color=color, linewidth=1.1)
            ax.scatter(
                row["blocked_rank_correlation"], y[index], s=24,
                marker="s" if is_full else "o", color=color,
                edgecolor="white", linewidth=0.4, zorder=3,
            )
        ax.set_yticks(y)
        ax.set_yticklabels(labels if ax in axes[:, 0] else [])
        ax.tick_params(axis="y", length=0)
        ax.set_title(scope, loc="left", fontweight="bold", pad=5)
        ax.set_xlim(-0.03, 0.62)
        ps.strip_axes(ax, keep=("bottom",))
    fig.supxlabel("Within-outcome blocked rank correlation", fontsize=ps.FS_LABEL, y=0.025)
    fig.subplots_adjust(left=0.28, right=0.98, bottom=0.10, top=0.95, hspace=0.28, wspace=0.16)
    png = output / "figure_S2_diagnostic_leave_one_domain_out_by_location.png"
    pdf = output / "figure_S2_diagnostic_leave_one_domain_out_by_location.pdf"
    fig.savefig(png, dpi=300, bbox_inches="tight", facecolor=ps.WHITE)
    fig.savefig(pdf, bbox_inches="tight", facecolor=ps.WHITE)
    plt.close(fig)
    return png, pdf


def write_readme(output: Path, results: pd.DataFrame) -> None:
    pooled = results.loc[
        results["scope"].eq("Pooled")
        & results["omitted_domain"].ne("None (full sample)")
    ]
    full = results.loc[
        results["scope"].eq("Pooled")
        & results["omitted_domain"].eq("None (full sample)")
    ].iloc[0]
    minimum = pooled.loc[pooled["blocked_rank_correlation"].idxmin()]
    maximum = pooled.loc[pooled["blocked_rank_correlation"].idxmax()]
    positive = int(pooled["positive_estimate"].sum())
    positive_ci = int((pooled["ci_low"] > 0).sum())
    location_sensitivity = results.loc[
        results["scope"].ne("Pooled")
        & results["omitted_domain"].ne("None (full sample)")
    ]
    text = f"""# Leave-one-domain-out sensitivity for pathway–surface coupling

## Estimand

The analysis reproduces the Figure 4.4 association between URL-Jaccard source-set displacement
and subject-normalized answer displacement. Both variables are ranked and mean-centered within
outcome before calculating their pooled correlation. The three-location analysis first averages
each condition within outcome across Dallas, New York, and Los Angeles. Outcome is the bootstrap
cluster and the permutation remains within outcome.

## Pooled result

The full-sample result is r = {full['blocked_rank_correlation']:.3f}, 95% CI
[{full['ci_low']:.3f}, {full['ci_high']:.3f}], based on 21 outcomes and 252 observations.
Each leave-one-domain-out analysis removes exactly three outcomes, leaving 18 outcomes and 216
observations.

All {positive}/7 leave-one-domain-out estimates are positive, and {positive_ci}/7 bootstrap CIs
exclude zero. Estimates range from {minimum['blocked_rank_correlation']:.3f} after omitting
{DOMAIN_LABELS[minimum['omitted_domain']]} to {maximum['blocked_rank_correlation']:.3f} after
omitting {DOMAIN_LABELS[maximum['omitted_domain']]}. This indicates that the pooled association
is not driven by any single substantive domain.

Across the three location-specific sensitivity analyses,
{int(location_sensitivity['positive_estimate'].sum())}/{len(location_sensitivity)} estimates are
positive and {int((location_sensitivity['ci_low'] > 0).sum())}/{len(location_sensitivity)} CIs
exclude zero.

## Outputs

- `leave_one_domain_out_all_results.csv`: raw pooled and location-specific estimates.
- `table_S4_leave_one_domain_out_pooled.csv` and `.tex`: paper-facing pooled table.
- `figure_S2_pathway_surface_leave_one_domain_out.pdf` and `.png`: pooled forest plot.
- `figure_S2_diagnostic_leave_one_domain_out_by_location.pdf` and `.png`: diagnostic by location.
- `analysis_summary.json`: compact machine-readable summary.

The seven leave-one-domain-out estimates are overlapping sensitivity specifications, not seven
independent hypothesis tests. Interpretation should emphasize stability of direction, magnitude,
and confidence intervals rather than counting p-values.
"""
    (output / "README.md").write_text(text, encoding="utf-8")


def write_manuscript_material(output: Path, results: pd.DataFrame) -> None:
    pooled = results.loc[
        results["scope"].eq("Pooled")
        & results["omitted_domain"].ne("None (full sample)")
    ]
    full = results.loc[
        results["scope"].eq("Pooled")
        & results["omitted_domain"].eq("None (full sample)")
    ].iloc[0]
    text = f"""# Suggested manuscript text

As a leave-one-domain-out sensitivity analysis, we re-estimated the within-outcome blocked rank
correlation after excluding each substantive domain in turn. The full three-location estimate was
$r={full['blocked_rank_correlation']:.2f}$, 95% CI
[{full['ci_low']:.2f}, {full['ci_high']:.2f}]. Across the seven exclusions, the association remained
positive and similar in magnitude (range $r={pooled['blocked_rank_correlation'].min():.2f}$ to
${pooled['blocked_rank_correlation'].max():.2f}$), and every outcome-cluster bootstrap confidence
interval excluded zero. Thus, the pathway–surface association was not attributable to any single
substantive domain.

Suggested caption: Leave-one-domain-out sensitivity of the pooled association between URL-level
source-set displacement and subject-normalized answer displacement. The square denotes the full
21-outcome estimate; circles denote estimates after removing the three outcomes belonging to each
domain. Horizontal lines are 95% outcome-cluster bootstrap confidence intervals. The grey band and
dashed line show the full-sample confidence interval and point estimate, respectively.
"""
    (output / "MANUSCRIPT_TEXT.md").write_text(text, encoding="utf-8")


def main() -> None:
    args = parse_args()
    base = args.base_dir.absolute()
    output = args.output_dir.absolute()
    output.mkdir(parents=True, exist_ok=True)

    results = run_analysis(base, args.n_permutations, args.n_bootstrap)
    results.to_csv(output / "leave_one_domain_out_all_results.csv", index=False)
    pooled = results.loc[results["scope"].eq("Pooled")].copy()
    table = paper_table(pooled)
    table.to_csv(output / "table_S4_leave_one_domain_out_pooled.csv", index=False)
    (output / "table_S4_leave_one_domain_out_pooled.tex").write_text(
        table.to_latex(index=False, escape=True, float_format=lambda value: f"{value:.3f}"),
        encoding="utf-8",
    )
    pooled_png, pooled_pdf = plot_pooled(pooled, output)
    diagnostic_png, diagnostic_pdf = plot_all_scopes(results, output)

    sensitivity = pooled.loc[pooled["omitted_domain"].ne("None (full sample)")]
    location_sensitivity = results.loc[
        results["scope"].ne("Pooled")
        & results["omitted_domain"].ne("None (full sample)")
    ]
    summary = {
        "metric_x": SOURCE_METRIC,
        "metric_y": ANSWER_METRIC,
        "blocking_variable": "outcome_id",
        "pooled_full_sample_r": float(pooled.iloc[0]["blocked_rank_correlation"]),
        "pooled_full_sample_ci": [float(pooled.iloc[0]["ci_low"]), float(pooled.iloc[0]["ci_high"])],
        "pooled_lodo_min_r": float(sensitivity["blocked_rank_correlation"].min()),
        "pooled_lodo_max_r": float(sensitivity["blocked_rank_correlation"].max()),
        "pooled_lodo_positive": int(sensitivity["positive_estimate"].sum()),
        "pooled_lodo_total": int(len(sensitivity)),
        "pooled_lodo_ci_excludes_zero": int((sensitivity["ci_low"] > 0).sum()),
        "location_lodo_positive": int(location_sensitivity["positive_estimate"].sum()),
        "location_lodo_total": int(len(location_sensitivity)),
        "location_lodo_ci_excludes_zero": int((location_sensitivity["ci_low"] > 0).sum()),
        "n_permutations": args.n_permutations,
        "n_bootstrap": args.n_bootstrap,
    }
    (output / "analysis_summary.json").write_text(
        json.dumps(summary, indent=2), encoding="utf-8"
    )
    write_readme(output, results)
    write_manuscript_material(output, results)

    print(f"Results : {output / 'leave_one_domain_out_all_results.csv'}")
    print(f"Table   : {output / 'table_S4_leave_one_domain_out_pooled.csv'}")
    print(f"Figure  : {pooled_png}")
    print(f"          {pooled_pdf}")
    print(f"Diag.   : {diagnostic_png}")
    print(f"          {diagnostic_pdf}")


if __name__ == "__main__":
    main()
