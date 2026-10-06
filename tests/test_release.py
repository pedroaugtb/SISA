"""Integrity and numerical smoke tests for the public reproducibility bundle."""

from __future__ import annotations

import ast
import csv
import json
import re
import struct
import unittest
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
COLLECTIONS = (
    ("annotations/v1_dallas", 273),
    ("annotations/v2_ny", 273),
    ("annotations/v3_la", 273),
    ("annotations/robustness/v1_dallas", 91),
    ("annotations/robustness/v2_ny", 91),
    ("annotations/robustness/v3_la", 91),
)


def read_csv(relative_path: str) -> list[dict[str, str]]:
    with (ROOT / relative_path).open(encoding="utf-8", newline="") as handle:
        return list(csv.DictReader(handle))


class CollectionIntegrityTests(unittest.TestCase):
    def test_collection_and_manifest_counts(self) -> None:
        for relative_root, expected in COLLECTIONS:
            with self.subTest(collection=relative_root):
                collection_root = ROOT / relative_root / "google_aio_collection"
                records = sorted(collection_root.glob("*/*.txt"))
                self.assertEqual(len(records), expected)

                manifest_path = collection_root / "query_manifest.csv"
                with manifest_path.open(encoding="utf-8", newline="") as handle:
                    rows = list(csv.DictReader(handle))
                self.assertEqual(len(rows), expected)
                self.assertEqual(len({row["query_id"] for row in rows}), expected)

                for row in rows:
                    record = collection_root.parent / row["file"]
                    self.assertTrue(record.is_file(), record)
                    text = record.read_text(encoding="utf-8")
                    self.assertRegex(text, r"(?m)^QUERY\s*$")
                    self.assertRegex(text, r"(?m)^AI OVERVIEW TEXT\s*$")

    def test_source_usage_points_to_collected_records(self) -> None:
        for relative_root, _ in COLLECTIONS:
            with self.subTest(collection=relative_root):
                version_root = ROOT / relative_root
                rows = read_csv(f"{relative_root}/source_corpus/url_usage.csv")
                self.assertTrue(rows)
                for row in rows:
                    record = version_root / "google_aio_collection" / row["file"]
                    self.assertTrue(record.is_file(), record)

    def test_no_nonblank_recorded_public_ip(self) -> None:
        pattern = re.compile(r"^Public IP:[ \t]*\S", re.MULTILINE)
        for relative_root, _ in COLLECTIONS:
            collection_root = ROOT / relative_root / "google_aio_collection"
            for record in collection_root.glob("*/*.txt"):
                self.assertIsNone(pattern.search(record.read_text(encoding="utf-8")), record)


class PublicSourceManifestTests(unittest.TestCase):
    def test_manifests_contain_only_safe_provenance(self) -> None:
        forbidden = {
            "headers",
            "http_headers",
            "request_headers",
            "response_headers",
            "raw_path",
            "text_path",
            "metadata_path",
            "traceback",
            "cookies",
        }
        digest = re.compile(r"^[0-9a-f]{64}$")

        for relative_root, _ in COLLECTIONS:
            manifest = ROOT / relative_root / "source_corpus" / "retrieval_manifest.csv"
            with self.subTest(manifest=manifest):
                with manifest.open(encoding="utf-8", newline="") as handle:
                    reader = csv.DictReader(handle)
                    self.assertTrue(reader.fieldnames)
                    self.assertTrue(forbidden.isdisjoint(set(reader.fieldnames or ())))
                    rows = list(reader)
                self.assertTrue(rows)
                for row in rows:
                    value = row["text_sha256"].strip()
                    if value:
                        self.assertRegex(value, digest)
                        self.assertTrue(row["text_filename"].strip())


class NumericalRegressionTests(unittest.TestCase):
    def test_pooled_aggregate_effects(self) -> None:
        rows = {
            row["analysis"]: row
            for row in read_csv(
                "results/three_location_pooled_secondary/pooled_aggregate_tests.csv"
            )
        }
        expected = {
            "citation_count": 2.5952380952380953,
            "source_url_jaccard_distance": 0.055712603994295735,
            "answer_semantic_displacement": 0.06355793916043781,
            "evidence_semantic_gap": -0.02030024264925316,
        }
        for analysis, value in expected.items():
            with self.subTest(analysis=analysis):
                self.assertAlmostEqual(float(rows[analysis]["mean_difference"]), value, places=12)

    def test_blocked_pipeline_correlations(self) -> None:
        rows = read_csv("figures_three_locations_pooled/pooled_pipeline_correlations.csv")
        keyed = {(row["metric_x"], row["metric_y"]): row for row in rows}

        source_answer = keyed[("url_jaccard_distance", "dense_distance_subject_normalized")]
        self.assertEqual(int(source_answer["n_observations"]), 252)
        self.assertAlmostEqual(
            float(source_answer["blocked_rank_correlation"]), 0.4010567820206754, places=12
        )

        evidence_answer = keyed[
            ("evidence_semantic_gap_source_balanced", "dense_distance_subject_normalized")
        ]
        self.assertEqual(int(evidence_answer["n_observations"]), 212)
        self.assertAlmostEqual(
            float(evidence_answer["blocked_rank_correlation"]), -0.38880328710837186, places=12
        )

    def test_grounding_eligibility_summary(self) -> None:
        path = ROOT / "supplementary_results/grounding_eligibility/analysis_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["all_primary_responses"], 819)
        self.assertEqual(summary["all_primary_eligible"], 476)
        self.assertEqual(summary["complete_location_outcome_pairs"], 122)
        self.assertAlmostEqual(
            summary["minority_minus_majority_rate_difference"],
            0.23015873015873017,
            places=12,
        )

    def test_supplied_annotation_agreement_and_disclosure(self) -> None:
        path = ROOT / "artifacts/annotation_results/agreement_summary.json"
        summary = json.loads(path.read_text(encoding="utf-8"))
        self.assertEqual(summary["full_pool"]["n"], 62)
        self.assertAlmostEqual(
            summary["full_pool"]["gwet_ac2_quadratic"], 0.644114801676879, places=12
        )
        disclosure = (ROOT / "KNOWN_DISCREPANCY.md").read_text(encoding="utf-8")
        self.assertIn("scale: 0.644", disclosure)
        self.assertIn("AC2 = 0.451", disclosure)


class ReleaseHygieneTests(unittest.TestCase):
    def test_main_figures_are_present(self) -> None:
        stems = (
            "figure_4_2_evidentiary_pathway_pooled3",
            "figure_4_3_semantic_shift_anatomy_pooled3",
            "figure_4_4_pathway_surface_coupling_pooled3",
            "figure_4_5_grounding_pooled3",
        )
        for stem in stems:
            for suffix in (".pdf", ".png"):
                path = ROOT / "figures_three_locations_pooled" / f"{stem}{suffix}"
                self.assertGreater(path.stat().st_size, 0)

    def test_python_sources_parse(self) -> None:
        for path in [ROOT / "run.py", *sorted((ROOT / "code").rglob("*.py"))]:
            with self.subTest(path=path):
                ast.parse(path.read_text(encoding="utf-8"), filename=str(path))

    def test_no_private_paths_or_cookie_headers_in_text_files(self) -> None:
        checked_suffixes = {".py", ".csv", ".json", ".md", ".tex", ".txt"}
        path_markers = (
            "/" + "scratch/",
            "/" + "home/grad/",
            "/" + "Users/",
            "C:" + "\\",
        )
        email_pattern = re.compile(
            r"(?<![\w./-])[A-Z0-9._%+-]+@[A-Z0-9.-]+\.[A-Z]{2,}(?![\w.-])",
            re.IGNORECASE,
        )
        for path in ROOT.rglob("*"):
            if not path.is_file() or path.suffix.lower() not in checked_suffixes:
                continue
            if path.relative_to(ROOT).parts[0] in {"reproduced_outputs", "reproduced_results"}:
                continue
            contents = path.read_text(encoding="utf-8", errors="replace")
            with self.subTest(path=path):
                for marker in path_markers:
                    self.assertNotIn(marker, contents)
                self.assertNotIn("Set-" + "Cookie:", contents)
                self.assertIsNone(email_pattern.search(contents))

    def test_binary_artifacts_have_no_identity_metadata(self) -> None:
        pdf_metadata = re.compile(
            rb"/(Author|Creator|Producer|CreationDate|ModDate)\s*(?:\(|<)",
            re.IGNORECASE,
        )
        for path in ROOT.rglob("*.pdf"):
            with self.subTest(path=path):
                self.assertIsNone(pdf_metadata.search(path.read_bytes()))

        for path in ROOT.rglob("*.png"):
            contents = path.read_bytes()
            with self.subTest(path=path):
                self.assertTrue(contents.startswith(b"\x89PNG\r\n\x1a\n"))
                offset = 8
                while offset + 12 <= len(contents):
                    length = struct.unpack(">I", contents[offset : offset + 4])[0]
                    chunk_type = contents[offset + 4 : offset + 8]
                    chunk_data = contents[offset + 8 : offset + 8 + length]
                    if chunk_type == b"tEXt":
                        self.assertTrue(
                            chunk_data.startswith(b"Software\x00Matplotlib version"),
                            f"unexpected PNG text metadata in {path}",
                        )
                    self.assertNotIn(
                        chunk_type,
                        {b"zTXt", b"iTXt", b"tIME"},
                        f"unexpected compressed text or timestamp metadata in {path}",
                    )
                    offset += length + 12

    def test_no_symbolic_links(self) -> None:
        self.assertEqual([path for path in ROOT.rglob("*") if path.is_symlink()], [])

    def test_github_file_size_limit(self) -> None:
        limit = 100 * 1024 * 1024
        oversized = [path for path in ROOT.rglob("*") if path.is_file() and path.stat().st_size >= limit]
        self.assertEqual(oversized, [])


if __name__ == "__main__":
    unittest.main()
