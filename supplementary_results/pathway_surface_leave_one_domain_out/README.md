# Leave-one-domain-out sensitivity for pathway–surface coupling

## Estimand

The analysis reproduces the Figure 4.4 association between URL-Jaccard source-set displacement
and subject-normalized answer displacement. Both variables are ranked and mean-centered within
outcome before calculating their pooled correlation. The three-location analysis first averages
each condition within outcome across Dallas, New York, and Los Angeles. Outcome is the bootstrap
cluster and the permutation remains within outcome.

## Pooled result

The full-sample result is r = 0.401, 95% CI
[0.282, 0.507], based on 21 outcomes and 252 observations.
Each leave-one-domain-out analysis removes exactly three outcomes, leaving 18 outcomes and 216
observations.

All 7/7 leave-one-domain-out estimates are positive, and 7/7 bootstrap CIs
exclude zero. Estimates range from 0.377 after omitting
Housing to 0.426 after
omitting Credit & financial services. This indicates that the pooled association
is not driven by any single substantive domain.

Across the three location-specific sensitivity analyses,
21/21 estimates are
positive and 21/21 CIs
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
