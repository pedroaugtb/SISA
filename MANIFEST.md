# Release manifest

## Included

- All 819 primary and 273 additional-outcome AIO observations.
- Query manifests and the manual collection protocol.
- Both outcome-rating files, matched agreement table, and primary/robustness selections.
- Safe source-retrieval provenance and query-to-source mappings for all locations and both outcome sets.
- One canonical copy of each active analysis, figure, supplement, and sensitivity script.
- Minimal per-location analysis-ready tables and JSON summaries required by downstream analyses.
- Inputs and summaries for label-normalization, chunking, and BGE encoder sensitivity.
- Main figures, supplementary figures/tables, and outcome-selection robustness outputs.
- Pinned Python package versions and embedding-model revisions.

## Excluded as unnecessary or unsuitable for GitHub

- `legacy/` and every abandoned analysis: hedging, veridicality, claim survival, source-authority taxonomy, DIF/random-slope pilots, and exploratory variance decomposition.
- Superseded Dallas-only figures and location-stratified presentation variants not used in the paper.
- Duplicate copies of the primary pipeline.
- Run logs, caches, downloaded models, notebook checkpoints, and temporary validation reports.
- Intermediate document embeddings, chunk dumps, passage-match dumps, and duplicated aggregate outputs when a smaller canonical table is sufficient.
- Raw cited webpages and full extracted page text. These are third-party works, include unnecessary HTTP headers/cookies in the working corpus, and occupy roughly 1.4 GB. Safe manifests, text hashes, frozen query-level metrics, and the re-fetch code are included instead.
- The anonymous review PDF, because the manuscript itself prohibits public sharing during review and is not required to run the analysis.

## Privacy and portability changes

- One recorded VPN IP was blanked from the public copy.
- One unnecessary third-party contact email embedded in an evidence excerpt was removed.
- Cookie-bearing HTTP headers, local absolute paths, and downloader tracebacks were removed from source provenance.
- Archived JSON path fields were converted to repository-relative paths.
- Script defaults now resolve the repository root from each script location rather than a private filesystem path.

No query, AIO answer, cited URL, experimental label, outcome rating, or numerical analysis value was altered.
