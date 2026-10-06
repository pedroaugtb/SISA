#!/usr/bin/env python3
"""Run evidence-alignment chunking sensitivity without touching primary outputs."""

from __future__ import annotations

import argparse
import json
import os
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

# The primary multilingual MPNet encoder has a 128-token SentenceTransformers
# limit. The production pipeline reserves 16 tokens, so source windows above
# 112 would be silently capped. The upper configuration therefore increases
# answer context and boundary redundancy while retaining the maximum defensible
# source window under the same encoder.
CONFIGURATIONS = {
    "smaller": {
        "source_chunk_tokens": 80,
        "source_chunk_overlap": 30,
        "aio_chunk_tokens": 64,
        "aio_chunk_overlap": 20,
        "description": "Smaller windows with more potential chunk boundaries.",
    },
    "high_context": {
        "source_chunk_tokens": 112,
        "source_chunk_overlap": 56,
        "aio_chunk_tokens": 112,
        "aio_chunk_overlap": 32,
        "description": (
            "Maximum same-encoder source window, greater source overlap, and "
            "a larger answer window."
        ),
    },
}


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--base-dir", type=Path, default=BASE_DEFAULT)
    parser.add_argument("--python", default=sys.executable)
    parser.add_argument(
        "--model",
        default="sentence-transformers/paraphrase-multilingual-mpnet-base-v2",
    )
    parser.add_argument("--device", choices=["cuda", "cpu"], default="cuda")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument(
        "--configurations",
        nargs="+",
        choices=CONFIGURATIONS,
        default=list(CONFIGURATIONS),
    )
    parser.add_argument(
        "--locations", nargs="+", choices=LOCATIONS, default=list(LOCATIONS)
    )
    parser.add_argument("--analysis-only", action="store_true")
    parser.add_argument("--skip-comparison", action="store_true")
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
    evidence_script = base / "code/analysis" / "evidence_synthesis_analysis.py"
    comparison_script = Path(__file__).with_name("analyze_chunking_sensitivity.py")
    sensitivity_root = base / "sensitivity_results/chunking_run"
    logs = sensitivity_root / "run_logs"
    sensitivity_root.mkdir(parents=True, exist_ok=True)

    manifest = {
        "created_at_utc": datetime.now(timezone.utc).isoformat(),
        "model": args.model,
        "device": args.device,
        "batch_size": args.batch_size,
        "locations": args.locations,
        "requested_configurations": args.configurations,
        "primary": {
            "source_chunk_tokens": 112,
            "source_chunk_overlap": 40,
            "aio_chunk_tokens": 96,
            "aio_chunk_overlap": 24,
            "note": "Effective runtime configuration after the model-length cap.",
        },
        "alternatives": CONFIGURATIONS,
        "primary_results_root": str(base / "results"),
        "eligibility": "Frozen from primary: fetch coverage >= .50 and >= 3 sources.",
    }
    (sensitivity_root / "run_configuration.json").write_text(
        json.dumps(manifest, indent=2, ensure_ascii=False) + "\n", encoding="utf-8"
    )

    if not args.analysis_only:
        for config_name in args.configurations:
            config = CONFIGURATIONS[config_name]
            raw_root = sensitivity_root / config_name / "raw_results"
            raw_root.mkdir(parents=True, exist_ok=True)
            for location in args.locations:
                semantic_file = (
                    base
                    / "results"
                    / f"semantic_embedding_analysis_{location}"
                    / "group_vs_generic_semantic_metrics.csv"
                )
                if not semantic_file.exists():
                    raise FileNotFoundError(
                        f"Primary semantic output is required: {semantic_file}"
                    )
                env = os.environ.copy()
                env.update(
                    {
                        "AIO_BASE_DIR": str(base),
                        "AIO_COLLECTION_VERSION": LOCATIONS[location],
                        "AIO_LOCATION_SLUG": location,
                        "AIO_RESULTS_ROOT": str(raw_root),
                        "AIO_SOURCE_OVERLAP_RESULTS_ROOT": str(base / "results"),
                        "AIO_SEMANTIC_RESULTS_ROOT": str(base / "results"),
                        "AIO_EMBEDDING_MODEL": args.model,
                        "AIO_EMBEDDING_DEVICE": args.device,
                        "AIO_EMBEDDING_BATCH_SIZE": str(args.batch_size),
                        "AIO_EMBEDDING_QUERY_PREFIX": "",
                        "AIO_EMBEDDING_TYPE": f"chunking sensitivity: {config_name}",
                        "AIO_SOURCE_CHUNK_TOKENS": str(config["source_chunk_tokens"]),
                        "AIO_SOURCE_CHUNK_OVERLAP": str(config["source_chunk_overlap"]),
                        "AIO_AIO_CHUNK_TOKENS": str(config["aio_chunk_tokens"]),
                        "AIO_AIO_CHUNK_OVERLAP": str(config["aio_chunk_overlap"]),
                        # Raw-run inference is not used in the cross-configuration
                        # sensitivity. Keep it lightweight; the comparator below
                        # performs the planned 20k/5k blocked analysis and effect
                        # tests on the frozen primary-eligible sample.
                        "AIO_N_PERMUTATIONS": "1000",
                        "AIO_N_BOOTSTRAP": "200",
                        "AIO_EXACT_SIGNFLIP_MAX_N": "0",
                        "AIO_N_BLOCK_PERMUTATIONS": "1000",
                        "AIO_N_BLOCK_BOOTSTRAP": "200",
                        "TOKENIZERS_PARALLELISM": "false",
                    }
                )
                run_logged(
                    [args.python, str(evidence_script)],
                    env,
                    logs / f"{config_name}_{location}.log",
                )

    if args.skip_comparison:
        print("Chunking reruns complete; comparison skipped:", sensitivity_root)
        return

    required = []
    for config_name in CONFIGURATIONS:
        for location in LOCATIONS:
            required.append(
                sensitivity_root
                / config_name
                / "raw_results"
                / f"evidence_synthesis_analysis_{location}_v2"
                / "mechanism_map_v2.csv"
            )
    missing = [path for path in required if not path.exists()]
    if missing:
        print("Comparison deferred because the following outputs are missing:")
        for path in missing:
            print(" -", path)
        print("After both configurations finish in all locations, rerun with --analysis-only.")
        return

    command = [
        args.python,
        str(comparison_script),
        "--base-dir",
        str(base),
        "--sensitivity-root",
        str(sensitivity_root),
        "--output-dir",
        str(sensitivity_root / "comparison"),
        "--n-bootstrap",
        str(args.n_bootstrap),
        "--n-permutations",
        str(args.n_permutations),
    ]
    run_logged(command, os.environ.copy(), logs / "comparison.log")
    print("Chunking sensitivity complete:", sensitivity_root)


if __name__ == "__main__":
    main()
