# Data dictionary

## Collected AIO responses

Each `annotations/**/google_aio_collection/<group>/<domain>__<outcome>.txt` file contains:

- `QUERY`: the exact query entered in Google Search;
- `LINKS`: cited or displayed supporting URLs, in displayed order;
- `AI OVERVIEW TEXT`: the complete copied response;
- `COLLECTION INFO`: available session notes;
- `METADATA`: query ID, social dimension, condition, group, domain, and outcome.

`query_manifest.csv` provides one row per expected query with the same identifiers. Primary manifests contain 273 rows; robustness manifests contain 91 rows.

One VPN IP value present in the working collection was blanked because it is not analytically necessary. One third-party contact email embedded in a cited evidence excerpt was also removed. Query text, AIO text, links, experimental metadata, and numerical analysis values were not changed.

## Outcome selection

`artifacts/annotation_inputs/outcomes_annotator{1,2}.csv` contain the two independent outcome ratings. The relevant columns are `Domain`, `Outcome`, the 1–5 star relevance item, and the binary comparability concern.

`artifacts/annotation_results/annotator_agreement_full.csv` is the matched 62-outcome analysis table. `selected_top3_outcomes.csv` contains the 21 primary outcomes. `annotations/robustness/selected_robustness_outcomes.csv` contains the next-ranked outcome in each domain.

## Cited-source provenance

Each `source_corpus/retrieval_manifest.csv` contains a safe subset of the original fetch manifest:

- URL identity: `source_id`, `normalized_url`, `registrable_domain`, `final_url`;
- retrieval: `fetch_timestamp_utc`, `success`, `status`, `http_status`, `downloaded_bytes`, `curl_returncode`;
- extraction: `extraction_status`, `text_chars`, `text_words`, `content_kind`, `content_type`;
- page metadata: title, author, date, site name, and canonical URL;
- verification: `text_filename` and `text_sha256` for extracted text that existed in the frozen corpus.

`url_usage.csv` maps each source to the AIO observation in which it appeared. Raw HTML/PDF files, extracted full text, HTTP headers/cookies, machine paths, and downloader tracebacks are excluded.

## Primary analysis-ready results

For each location (`dallas`, `ny`, `la`):

- `link_count_analysis_*/clean_aio_link_data.csv`: response-level citation counts;
- `source_overlap_analysis_*/source_overlap_vs_generic.csv`: URL/hostname displacement and directional coverage;
- `semantic_embedding_analysis_*/group_vs_generic_semantic_metrics.csv`: answer and sentence-level displacement metrics;
- `evidence_synthesis_analysis_*_v2/query_evidence_alignment_metrics_v2.csv`: response-level source recovery and evidence-alignment metrics;
- `evidence_synthesis_analysis_*_v2/mechanism_map_v2.csv`: merged pathway, answer, and evidence measures used for blocked associations;
- `all_results*.json`: per-location inferential results used by the three-location summary.

`results/three_location_pooled_secondary/` contains outcome-by-location effects plus pooled dimension and aggregate tests. `figures_three_locations_pooled/` contains the exact plotted values and the four pooled main figures.

## Robustness and sensitivity

`robustness_results/` mirrors the primary result layout for the seven additional outcomes. `paper_outputs/` contains the outcome-selection robustness figure and tables.

`sensitivity_results/label_normalization/` compares raw and label-normalized answer embeddings. `sensitivity_results/chunking/` contains comparison summaries plus the six alternative mechanism maps needed to rerun the comparison. `sensitivity_results/encoder_bge_large/` contains BGE comparison summaries and the minimal alternative-encoder metrics needed to rerun them.

`supplementary_results/` contains leave-one-domain-out, leave-one-location-out, outcome heterogeneity/influence, and evidence-eligibility outputs.
