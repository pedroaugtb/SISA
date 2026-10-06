# Grounding eligibility and effective sample size

This audit uses the exact reference rule and primary inputs used by Figure 4.5. It distinguishes
response-level eligibility, location-specific complete focal--comparison pairs, pooled
dimension--outcome pairs, and the condition--outcome means entering the continuous grounding
association.

Headline counts:

- all primary responses: 476/819 eligible (58.1%);
- explicit minority/majority responses: 441/756 eligible (58.3%);
- control responses: 35/63 eligible (55.6%);
- complete location-specific focal--comparison pairs: 122/378 (32.3%);
- pooled dimension--outcome pairs with at least one complete location: 76/126;
- condition--outcome means in the Figure 4.5 continuous association: 212.

The 212 condition--outcome means comprise
115 minority and
97 majority observations;
67 are based on one eligible location,
61 on two, and 84 on all three.

The condition comparison uses outcome, rather than response-location, as the inferential unit.
Location-specific binary eligibility indicators are averaged within outcome and condition before
the paired sign-flip test and outcome bootstrap. Dimension-specific p-values are Holm-adjusted;
the overall contrast is the prespecified summary and is reported separately.

Outputs include the paper-facing Table S8, full response and pair inventories, condition tests,
criterion-level diagnostics, `analysis_summary.json`, and suggested manuscript language.
