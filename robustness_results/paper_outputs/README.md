# Outcome-selection robustness: paper outputs

The main figure compares the 21 primary outcomes with the seven additional robustness
outcomes for citation count, source URL displacement, and answer semantic displacement.
The supplementary Figure S1 adds evidence semantic gap. Figures contain no overall title or
caption; those should be supplied in LaTeX.

For Figure S1, the evidence-gap panel has 11–14 eligible primary outcomes and 3–4 eligible
robustness outcomes across social dimensions. The pooled aggregate uses 18/21 primary outcomes
and 3/7 robustness outcomes. These counts belong in the caption rather than inside the panel.

Colors and typography follow `code/shared/figure_common.py`. Blue circles
represent the primary outcome set and orange diamonds represent the robustness outcome set.
Horizontal lines are 95% paired-outcome bootstrap confidence intervals.

Files:

- `figure_outcome_selection_robustness.{pdf,png}`: three-panel main-paper figure.
- `figure_outcome_selection_robustness_semantic_anatomy_style.{pdf,png}`: alternative
  three-panel version following the directional visual grammar of Figure 4.3.
- `figure_S1_outcome_selection_robustness.{pdf,png}`: four-panel supplementary figure.
- `table_S1_robustness_agreement.{csv,tex}`: directional and CI agreement summary.
- `table_S2_location_dimension_effects.{csv,tex}`: all metric × location × dimension effects.
- `table_S3_eligible_outcomes.{csv,tex}`: exact eligible and possible outcome counts.
- `figure_outcome_selection_robustness_stats.csv`: all values plotted in both figures.

Regenerate from the repository root with:

```bash
python run.py supplement
```
