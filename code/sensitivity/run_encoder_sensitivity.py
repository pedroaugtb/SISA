#!/usr/bin/env python3
"""Run the three-location encoder sensitivity without touching primary outputs."""

from __future__ import annotations

import argparse
import hashlib
import json
import os
import re
import subprocess
import sys
from datetime import datetime, timezone
from pathlib import Path


BASE_DEFAULT = Path(__file__).resolve().parents[2]
LOCATIONS = {
    "dallas": "v1_dallas",
    "ny": "v2_ny",
    "la": "v3_la",
}


def slugify(value: str) -> str:
    name = Path(value).name or value
    slug = re.sub(r"[^a-zA-Z0-9._-]+", "-", name).strip("-").lower()
    return slug or hashlib.sha256(value.encode()).hexdigest()[:12]


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument("--model", default="BAAI/bge-large-en-v1.5")
    parser.add_argument(
        "--revision",
        default="d4aa6901d3a41ba39fb536a557fa166f842b0e09",
        help="Pinned Hugging Face revision; pass an empty string for a local path.",
    )
    parser.add_argument("--model-slug")
    parser.add_argument(
        "--query-prefix",
        default="Represent this sentence for searching relevant passages: ",
        help="Applied only when ranking source chunks for a query.",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--locations", nargs="+", choices=LOCATIONS, default=list(LOCATIONS))
    parser.add_argument("--skip-semantic", action="store_true")
    parser.add_argument("--skip-evidence", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--n-bootstrap", type=int, default=5000)
    parser.add_argument("--n-permutations", type=int, default=20000)
    return parser.parse_args()


def run_logged(command: list[str], env: dict[str, str], log_path: Path) -> None:
    log_path.parent.mkdir(parents=True, exist_ok=True)
    print("RUNNING:", " ".join(command), flush=True)
    print("LOG:", log_path, flush=True)
    with log_path.open("w", encoding="utf-8") as handle:
        process = subprocess.Popen(
            command,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            text=True,
            bufsize=1,
        )
        assert process.stdout is not None
        for line in process.stdout:
            print(line, end="")
            handle.write(line)
            handle.flush()
        return_code = process.wait()
    if return_code:
        raise subprocess.CalledProcessError(return_code, command)


def main() -> None:
    args = parse_args()
    base = args.base_dir.resolve()
    scripts = base / "code/analysis"
    comparison_script = Path(__file__).with_name("analyze_encoder_sensitivity.py")
    model_slug = args.model_slug or slugify(args.model)
    sensitivity_root = base / "sensitivity_results/encoder_runs" / model_slug
    raw_root = sensitivity_root / "raw_results"
    logs = sensitivity_root / "run_logs"
    raw_root.mkdir(parents=True, exist_ok=True)

    config = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "model_revision": args.revision,
        "model_slug": model_slug,
        "query_prefix": args.query_prefix,
        "device": args.device,
        "batch_size": args.batch_size,
        "locations": args.locations,
        "source_chunk_tokens": 112,
        "source_chunk_overlap": 40,
        "aio_chunk_tokens": 96,
        "aio_chunk_overlap": 24,
        "label_normalization": "same as primary pipeline",
        "multilingual_responses_excluded": False,
        "primary_results_root": str(base / "results"),
        "alternative_results_root": str(raw_root),
    }
    (sensitivity_root / "run_configuration.json").write_text(
        json.dumps(config, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.analysis_only:
        for location in args.locations:
            version = LOCATIONS[location]
            env = os.environ.copy()
            env.update(
                {
                    "AIO_BASE_DIR": str(base),
                    "AIO_COLLECTION_VERSION": version,
                    "AIO_LOCATION_SLUG": location,
                    "AIO_RESULTS_ROOT": str(raw_root),
                    "AIO_SOURCE_OVERLAP_RESULTS_ROOT": str(base / "results"),
                    "AIO_SEMANTIC_RESULTS_ROOT": str(raw_root),
                    "AIO_EMBEDDING_MODEL": args.model,
                    "AIO_EMBEDDING_REVISION": args.revision,
                    "AIO_EMBEDDING_DEVICE": args.device,
                    "AIO_EMBEDDING_BATCH_SIZE": str(args.batch_size),
                    "AIO_EMBEDDING_QUERY_PREFIX": args.query_prefix,
                    "AIO_EMBEDDING_TYPE": "encoder sensitivity",
                    # Keep the actual reference configuration fixed. The original
                    # multilingual MPNet capped the nominal 160-token source window
                    # at 112 tokens because its model maximum was 128.
                    "AIO_SOURCE_CHUNK_TOKENS": "112",
                    "AIO_SOURCE_CHUNK_OVERLAP": "40",
                    "AIO_AIO_CHUNK_TOKENS": "96",
                    "AIO_AIO_CHUNK_OVERLAP": "24",
                    "TOKENIZERS_PARALLELISM": "false",
                }
            )
            if not args.skip_semantic:
                run_logged(
                    [args.python, str(scripts / "semantic_displacement_analysis.py")],
                    env,
                    logs / f"{location}_semantic_displacement.log",
                )
            if not args.skip_evidence:
                semantic_file = (
                    raw_root
                    / f"semantic_embedding_analysis_{location}"
                    / "group_vs_generic_semantic_metrics.csv"
                )
                if not semantic_file.exists():
                    raise FileNotFoundError(
                        f"Alternative semantic output is required before evidence: {semantic_file}"
                    )
                run_logged(
                    [args.python, str(scripts / "evidence_synthesis_analysis.py")],
                    env,
                    logs / f"{location}_evidence_alignment.log",
                )

    if args.skip_comparison:
        print("Encoder runs complete; comparison skipped:", sensitivity_root)
        return

    required = []
    for location in LOCATIONS:
        required.extend(
            [
                raw_root
                / f"semantic_embedding_analysis_{location}"
                / "group_vs_generic_semantic_metrics.csv",
                raw_root
                / f"evidence_synthesis_analysis_{location}_v2"
                / "query_evidence_alignment_metrics_v2.csv",
            ]
        )
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Comparison deferred because the following outputs are missing:")
        for path in missing:
            print(" -", path)
        print("After all locations finish, rerun with --analysis-only.")
        return

    command = [
        args.python,
        str(comparison_script),
        "--base-dir",
        str(base),
        "--alternative-root",
        str(raw_root),
        "--output-dir",
        str(sensitivity_root / "comparison"),
        "--model-name",
        args.model,
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--n-permutations",
        str(args.n_permutations),
    ]
    run_logged(command, os.environ.copy(), logs / "comparison.log")
    print("Encoder sensitivity complete:", sensitivity_root)


if __name__ == "__main__":
    main()
