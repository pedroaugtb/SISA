# Reproducibility Package for the Anonymous Submission

This repository is the minimal reproducibility package for the paper. It contains the 1,092 manually collected Google AI Overview observations, the two outcome-selection annotations, safe cited-source provenance, the code used for every analysis reported in the manuscript, exact analysis-ready metrics, and the paper figures and supplementary tables.

The package intentionally excludes abandoned analyses, legacy Dallas-only outputs, duplicate script trees, logs, model caches, intermediate embeddings, raw downloaded webpages, and the anonymous review PDF.



## Quick start

Python 3.12.3 was used for the archived results. From the repository root:

```bash
python3 -m venv .venv
source .venv/bin/activate
python -m pip install -r requirements.txt
python run.py validate
python -m unittest discover -s tests -v
python run.py all-derived
sha256sum -c CHECKSUMS.sha256
```

`validate` checks all six collections: 273 primary observations and 91 robustness observations in each of Dallas, New York, and Los Angeles. The unit tests check file integrity, privacy/portability constraints, and key reported numerical values. `all-derived` regenerates the pooled main figures, supplementary analyses, robustness figure/tables, and sensitivity comparisons from the archived analysis-ready CSV files. Generated files are written under `reproduced_outputs/` and are ignored by Git.

GPU acceleration is recommended only for rerunning embeddings and evidence alignment. Validation, pooled statistics, figures, and sensitivity comparisons run on CPU.

## Reproduction levels

The repository supports two distinct levels of reproduction.

### Exact paper statistics and figures

The checked-in `results/`, `robustness_results/`, and `sensitivity_results/` tables are the frozen analysis-ready values used by the paper. Regenerate all downstream statistics and graphics with:

```bash
python run.py all-derived
```

Run individual groups with `figures`, `supplement`, or `sensitivity` instead of `all-derived`.

### Recompute from collected AIO responses

The following reruns validation, citation counts, URL/host composition, and answer-semantic analyses into `reproduced_results/`:

```bash
python run.py analysis --dataset both
```

Restrict by dataset, location, or stage, for example:

```bash
python run.py analysis --dataset primary --locations dallas --only source_overlap
```

The answer-displacement stage downloads the pinned Hugging Face model on first use.

Evidence alignment additionally needs cited webpage text. Full third-party webpages are not redistributed because of copyright, privacy, and repository-size concerns. To build a fresh local source corpus and rerun alignment:

```bash
python run.py analysis --dataset primary --download-sources --include-evidence
python run.py analysis --dataset robustness --download-sources --include-evidence
```

Web content and availability can change. A fresh retrieval therefore may not exactly match the September–October 2026 source snapshot. The frozen query-level alignment metrics are included so every paper statistic remains exactly reproducible. Each public retrieval manifest records the original URL, status, extraction counts, timestamp, and SHA-256 hash of successfully extracted text.

## Design and sample

The query template was:

```text
What factors influence [OUTCOME] for [GROUP] in [DOMAIN]?
```

The primary design crosses 21 outcomes in seven domains with 13 group conditions, producing 273 queries per location and 819 observations overall. The outcome-selection robustness set contains seven additional outcomes, again crossed with 13 conditions and three locations, producing 273 further observations.

The six matched social dimensions are race, ethnicity, gender, disability, sexual orientation, and gender identity. Each has a minority-marked condition, a majority-marked condition, and the shared generic `people` condition.

Collection followed `annotations/collection_protocol.txt`: Chrome Guest mode, no signed-in Google account, a fresh session per query, and Windscribe endpoints in Dallas, New York, and Los Angeles.

## Statistical pipeline

- Citation volume: paired Wilcoxon signed-rank tests, paired rank-biserial effects, and 20,000 outcome-bootstrap samples.
- Source-set displacement: normalized URL and hostname Jaccard distance from the matched generic response, paired sign-flip tests, and Holm correction across six dimensions.
- Answer displacement: subject-label-normalized `all-mpnet-base-v2` embeddings, with raw text, TF–IDF, sentence coverage, and novelty checks.
- Evidence alignment: query-relevant passages from recoverable sources, source-balanced similarity using multilingual MPNet, at least 50% source recovery and at least three usable sources.
- Pipeline associations: within-outcome centered rank correlations, 20,000 within-outcome permutations, and 5,000 outcome-cluster bootstrap samples.
- Cross-location figures: location-specific effects are averaged within outcome before inference; locations are not treated as independent outcomes.

The pinned model revisions are:

- `sentence-transformers/all-mpnet-base-v2@e8c3b32edf5434bc2275fc9bab85f82640a19130`
- `sentence-transformers/paraphrase-multilingual-mpnet-base-v2@4328cf26390c98c5e3c738b4460a05b95f4911f5`
- `BAAI/bge-large-en-v1.5@d4aa6901d3a41ba39fb536a557fa166f842b0e09`

## Repository map

```text
artifacts/                   outcome ratings, agreement results, selected outcomes
code/analysis/               primary computational pipeline
code/query_design/           primary and robustness query generation
code/figures/                pooled main and outcome-distribution figures
code/supplement/             outcome robustness, leave-one-out, and eligibility analyses
code/sensitivity/            normalization, encoder, and chunking sensitivity
results/                     frozen primary analysis-ready results
robustness_results/          frozen additional-outcome results and paper outputs
sensitivity_results/         frozen comparison inputs and sensitivity summaries
supplementary_results/       supplementary tables and figures
figures_three_locations_pooled/ main pooled figures and plotted statistics
run.py                       portable command-line entry point
```

See `DATA_DICTIONARY.md` for file-level descriptions and `MANIFEST.md` for the inclusion/exclusion rationale.

## Important discrepancy to resolve

`KNOWN_DISCREPANCY.md` documents one mismatch between the manuscript and supplied outcome-rating files. The current CSVs reproduce relevance AC2 = 0.644 on the manuscript’s stated 0–5 scale, not 0.451. Resolve that item before public release; all main AIO audit results reproduce as reported.

## Before final public release

The supplied manuscript states that it is an anonymized submission whose non-anonymous public sharing is prohibited during review. During review, keep the source repository private and distribute only the anonymous proxy URL. When the venue permits identification, complete `PUBLICATION_CHECKLIST.md`, including author metadata, citation, DOI, repository license, and any data-use notices.
