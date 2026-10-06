#!/usr/bin/env python3
"""Portable entry point for the paper's reproducibility workflows."""

from __future__ import annotations

import argparse
import csv
import os
import subprocess
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parent
ANALYSIS = ROOT / "code" / "analysis"
FIGURES = ROOT / "code" / "figures"
SUPPLEMENT = ROOT / "code" / "supplement"
SENSITIVITY = ROOT / "code" / "sensitivity"

LOCATIONS = {
    "dallas": ("v1_dallas", "Dallas, Texas"),
    "ny": ("v2_ny", "New York City, New York"),
    "la": ("v3_la", "Los Angeles, California"),
}

PIPELINE = (
    ("link_count", "link_count_analysis.py"),
    ("source_overlap", "source_overlap_analysis.py"),
    ("semantic_displacement", "semantic_displacement_analysis.py"),
    ("source_download", "collect_cited_sources.py"),
    ("evidence_alignment", "evidence_synthesis_analysis.py"),
)


def execute(command: list[str], env: dict[str, str] | None = None) -> None:
    print("\nRUNNING:", " ".join(command), flush=True)
    subprocess.run(command, cwd=ROOT, env=env, check=True)


def collection_version(dataset: str, version: str) -> str:
    return version if dataset == "primary" else f"robustness/{version}"


def validate_collection(python: str, dataset: str, slug: str) -> None:
    version, _ = LOCATIONS[slug]
    version = collection_version(dataset, version)
    collection = ROOT / "annotations" / version / "google_aio_collection"
    expected = 273 if dataset == "primary" else 91
    execute(
        [
            python,
            str(ANALYSIS / "consistency.py"),
            "--collection-dir",
            str(collection),
        ]
    )
    report = collection / "consistency_report.csv"
    with report.open(encoding="utf-8", newline="") as handle:
        rows = list(csv.DictReader(handle))
    errors = [row for row in rows if row.get("status", "").upper() == "ERROR"]
    if len(rows) != expected or errors:
        raise RuntimeError(
            f"{dataset}/{slug}: expected {expected} valid records; "
            f"found {len(rows)} records and {len(errors)} errors"
        )


def run_analysis(args: argparse.Namespace) -> None:
    datasets = [args.dataset] if args.dataset != "both" else ["primary", "robustness"]
    for dataset in datasets:
        expected_outcomes = 21 if dataset == "primary" else 7
        results_root = ROOT / "reproduced_results" / dataset
        results_root.mkdir(parents=True, exist_ok=True)

        if dataset == "primary" and args.only is None:
            agreement_env = os.environ.copy()
            agreement_env.update(
                {
                    "AIO_BASE_DIR": str(ROOT),
                    "AIO_ANNOTATION_OUTPUT_DIR": str(
                        ROOT / "reproduced_results" / "query_selection"
                    ),
                }
            )
            execute(
                [args.python, str(ANALYSIS / "query_selection_agreement.py")],
                agreement_env,
            )

        for slug in args.locations:
            version, label = LOCATIONS[slug]
            version = collection_version(dataset, version)
            validate_collection(args.python, dataset, slug)
            if args.only == "consistency":
                continue

            env = os.environ.copy()
            env.update(
                {
                    "AIO_BASE_DIR": str(ROOT),
                    "AIO_COLLECTION_VERSION": version,
                    "AIO_LOCATION_SLUG": slug,
                    "AIO_LOCATION_LABEL": label,
                    "AIO_EXPECTED_OUTCOMES": str(expected_outcomes),
                    "AIO_RESULTS_ROOT": str(results_root),
                    "AIO_SOURCE_OVERLAP_RESULTS_ROOT": str(results_root),
                    "AIO_SEMANTIC_RESULTS_ROOT": str(results_root),
                }
            )

            for stage, filename in PIPELINE:
                if args.only and stage != args.only:
                    continue
                if stage == "source_download" and not args.download_sources:
                    continue
                if stage == "evidence_alignment" and not args.include_evidence:
                    continue
                if stage == "evidence_alignment":
                    corpus = ROOT / "annotations" / version / "source_corpus"
                    missing = [
                        path
                        for path in (corpus / "manifest.csv", corpus / "url_usage.csv")
                        if not path.exists()
                    ]
                    if missing:
                        raise FileNotFoundError(
                            "Evidence alignment needs a local source corpus. "
                            "Rerun with --download-sources first. Missing: "
                            + ", ".join(map(str, missing))
                        )
                execute([args.python, str(ANALYSIS / filename)], env)

        if args.only is None and set(args.locations) == set(LOCATIONS):
            if args.include_evidence:
                execute(
                    [
                        args.python,
                        str(ANALYSIS / "summarize_three_locations.py"),
                        "--base-dir",
                        str(ROOT),
                        "--results-root",
                        str(results_root),
                    ]
                )
            if args.include_evidence:
                execute(
                    [
                        args.python,
                        str(ANALYSIS / "pooled_three_location_secondary.py"),
                        "--base-dir",
                        str(ROOT),
                        "--results-root",
                        str(results_root),
                    ]
                )


def run_figures(python: str) -> None:
    output = ROOT / "reproduced_outputs" / "main_figures"
    execute(
        [
            python,
            str(FIGURES / "RUN_FIGURES.py"),
            "--base-dir",
            str(ROOT),
            "--output-dir",
            str(output),
            "--force-recompute",
        ]
    )


def run_supplement(python: str) -> None:
    output = ROOT / "reproduced_outputs" / "supplement"
    output.mkdir(parents=True, exist_ok=True)
    jobs = [
        ("analyze_pathway_surface_leave_one_domain_out.py", "pathway_surface_leave_one_domain_out"),
        ("analyze_pathway_surface_leave_one_location_out.py", "pathway_surface_leave_one_location_out"),
        ("analyze_outcome_effect_heterogeneity.py", "outcome_effect_heterogeneity"),
        ("analyze_grounding_eligibility.py", "grounding_eligibility"),
    ]
    for script, folder in jobs:
        execute(
            [
                python,
                str(SUPPLEMENT / script),
                "--base-dir",
                str(ROOT),
                "--output-dir",
                str(output / folder),
            ]
        )
    execute(
        [
            python,
            str(SUPPLEMENT / "plot_outcome_robustness.py"),
            "--base-dir",
            str(ROOT),
            "--output-dir",
            str(output / "outcome_selection_robustness"),
        ]
    )
    execute(
        [
            python,
            str(FIGURES / "S4_pooled.py"),
            "--results-dir",
            str(output / "outcome_effect_heterogeneity"),
            "--output-dir",
            str(output / "outcome_effect_heterogeneity"),
        ]
    )


def run_sensitivity_summaries(python: str) -> None:
    output = ROOT / "reproduced_outputs" / "sensitivity"
    output.mkdir(parents=True, exist_ok=True)
    execute(
        [
            python,
            str(SENSITIVITY / "analyze_label_normalization_sensitivity.py"),
            "--base-dir",
            str(ROOT),
            "--output-dir",
            str(output / "label_normalization"),
        ]
    )
    execute(
        [
            python,
            str(SENSITIVITY / "analyze_chunking_sensitivity.py"),
            "--base-dir",
            str(ROOT),
            "--sensitivity-root",
            str(ROOT / "sensitivity_results" / "chunking"),
            "--output-dir",
            str(output / "chunking"),
        ]
    )
    execute(
        [
            python,
            str(SENSITIVITY / "analyze_encoder_sensitivity.py"),
            "--base-dir",
            str(ROOT),
            "--alternative-root",
            str(ROOT / "sensitivity_results" / "encoder_bge_large" / "raw"),
            "--output-dir",
            str(output / "encoder_bge_large"),
            "--model-name",
            "BAAI/bge-large-en-v1.5@d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        ]
    )


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument(
        "command",
        choices=["validate", "analysis", "figures", "supplement", "sensitivity", "all-derived"],
    )
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--dataset", choices=["primary", "robustness", "both"], default="both")
    parser.add_argument("--locations", nargs="+", choices=LOCATIONS, default=list(LOCATIONS))
    parser.add_argument(
        "--only",
        choices=["consistency"] + [stage for stage, _ in PIPELINE],
    )
    parser.add_argument(
        "--download-sources",
        action="store_true",
        help="Re-fetch cited pages. Web content may have changed since the paper snapshot.",
    )
    parser.add_argument("--include-evidence", action="store_true")
    args = parser.parse_args()

    if args.command == "validate":
        for dataset in ([args.dataset] if args.dataset != "both" else ["primary", "robustness"]):
            for slug in args.locations:
                validate_collection(args.python, dataset, slug)
    elif args.command == "analysis":
        run_analysis(args)
    elif args.command == "figures":
        run_figures(args.python)
    elif args.command == "supplement":
        run_supplement(args.python)
    elif args.command == "sensitivity":
        run_sensitivity_summaries(args.python)
    else:
        run_figures(args.python)
        run_supplement(args.python)
        run_sensitivity_summaries(args.python)


if __name__ == "__main__":
    main()
