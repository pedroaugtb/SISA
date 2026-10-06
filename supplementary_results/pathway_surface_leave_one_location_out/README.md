# Leave-one-location-out sensitivity for pathway–surface coupling

The analysis reconstructs the three-location pooled mechanism map after dropping Dallas, New York,
or Los Angeles in turn. For every retained subset, source-set displacement and answer displacement
are averaged within the same condition–outcome across the retained locations before ranking and
mean-centering within outcome. Outcome remains the bootstrap cluster and permutations remain
within outcome.

The full pooled estimate is r = 0.401, 95% CI
[0.282, 0.507]. All 3/3 leave-one-location-out estimates are positive,
and all 3/3 confidence intervals exclude zero. Estimates range from
0.329 after omitting Los Angeles to
0.483 after omitting New York.

Each specification contains 21 outcomes and 252 condition–outcome observations. The sensitivity
specifications differ only in which geographic repetitions are averaged into those observations.

Outputs:

- `leave_one_location_out_results.csv`: raw estimates and uncertainty.
- `table_S5_leave_one_location_out.csv` and `.tex`: paper-facing table.
- `figure_S3_pathway_surface_leave_one_location_out.pdf` and `.png`: optional diagnostic figure.
- `MANUSCRIPT_TEXT.md`: suggested results text and caption.
