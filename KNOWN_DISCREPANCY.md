# Known manuscript/data discrepancy

The manuscript reports full-pool relevance agreement as quadratically weighted Gwet's AC2 = 0.451, 95% CI [0.240, 0.662], with 61.3% of ratings within one point.

The two supplied annotation CSVs contain 62 matched outcomes and reproduce:

- exact agreement: 25.8% (16/62), matching the manuscript;
- within-one agreement: 59.7% (37/62), not 61.3%;
- AC2 on the manuscript's stated 0–5 category scale: 0.644, 95% CI [0.512, 0.776];
- AC2 on the annotation form's displayed 1–5 scale: 0.421, 95% CI [0.203, 0.638].

The selected 21-outcome descriptives do reproduce the manuscript: 52.4% exact agreement, 95.2% within one point, mean absolute difference 0.52, and 95.2% comparability agreement. Binary comparability agreement also reproduces: AC1 = 0.966, 95% CI [0.916, 1.000]. The selected outcomes themselves are unchanged.

The available Git history contains only the current annotation files, so the earlier values cannot be reconstructed from this workspace. Before publication, do one of the following:

1. replace the rating CSVs with the exact frozen snapshot used for AC2 = 0.451 and 61.3%; or
2. update the manuscript to the values reproduced from the released data and clearly state whether the ordinal scale is 1–5 or 0–5.

Do not edit the code to force the manuscript value. The release should keep the statistic tied to an auditable data snapshot.
