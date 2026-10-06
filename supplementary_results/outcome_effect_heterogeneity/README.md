# Outcome-level effect heterogeneity

This analysis evaluates whether the pooled minority-minus-majority effects in the primary
21-outcome experiment are broadly distributed across outcomes or disproportionately influenced by
a small number of cases. It covers citation count, source URL displacement, and answer semantic
displacement.

For every measure, social dimension, and outcome, the contrast is first calculated separately in
Dallas, New York, and Los Angeles. The three location-specific contrasts are then averaged within
the same outcome. Locations are therefore repetitions, not independent outcomes. Every
measure-by-dimension distribution contains 21 equally weighted primary outcomes.

Across the 18 measure-by-dimension cells, 15/18 medians are positive and
15/18 have more than half of their outcomes above zero. The leave-one-outcome
analysis found at least one sign reversal in 0/18 cells (0 of 378
omissions in total).

The distribution differs by measure. Citation count has positive medians and a majority of
positive outcomes in 6/6 dimensions; answer semantic displacement
does so in 6/6. Source URL displacement has a positive median in
3/6 dimensions and a majority of positive outcomes in
3/6, reflecting the large number of exact zero contrasts visible
in Figure S4.

`positive`, `zero`, and `negative` use a numerical zero tolerance of 1e-12. Quartiles
use linear interpolation. This is a descriptive heterogeneity and influence analysis; it does not
introduce a new family of hypothesis tests.

Outputs:

- `outcome_effects_long.csv`: all 378 pooled outcome effects.
- `table_S6_outcome_heterogeneity.csv` and `.tex`: distribution summaries.
- `leave_one_outcome_out_all_results.csv`: all 378 influence diagnostics, including outcome names.
- `table_S7_leave_one_outcome_influence.csv` and `.tex`: compact influence summaries.
- `figure_S4_outcome_effect_distributions.pdf` and `.png`: outcome points, IQRs, medians, and means.
- `analysis_summary.json`: machine-readable headline diagnostics.
