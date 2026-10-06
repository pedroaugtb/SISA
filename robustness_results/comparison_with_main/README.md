# Comparison of the main and robustness outcome sets

The main analysis contains 21 outcomes. The robustness analysis contains seven new outcomes,
one per domain, measured in the same three locations. Effects are minority minus majority.

## Aggregate pooled effects

| Analysis | Main estimate [95% CI] | Robustness estimate [95% CI] | Same direction |
|---|---:|---:|:---:|
| Citation count | 2.5952 [1.9497, 3.1614] | 2.2698 [1.3333, 3.1905] | yes |
| Source URL displacement | 0.0557 [0.0361, 0.0772] | 0.0559 [0.0257, 0.0827] | yes |
| Answer semantic displacement | 0.0636 [0.0453, 0.0814] | 0.0770 [0.0612, 0.0922] | yes |
| Evidence semantic gap | -0.0203 [-0.0347, -0.0063] | -0.0146 [-0.0281, -0.0054] | yes |

All four aggregate effects retain their direction. Inferential precision is lower in the
robustness set because there are only seven outcomes. The evidence-semantic-gap aggregate
uses only three eligible outcomes and should therefore be treated as especially provisional.

## Consistency by social dimension

| Analysis | Pooled direction retained | Location-specific direction retained | Robustness CIs excluding zero |
|---|---:|---:|---:|
| Citation count | 6/6 | 18/18 | 2/6 |
| Source URL displacement | 6/6 | 16/18 | 4/6 |
| Answer semantic displacement | 6/6 | 16/18 | 4/6 |
| Evidence semantic gap | 5/6 | 13/18 | 3/6 |

The p-values and Holm decisions are not directly comparable as evidence of replication because
the robustness analysis has one third as many outcomes. Direction, magnitude, uncertainty, and
coverage are the primary comparison targets.

## Source-text coverage

Text extraction rates for unique source URLs were Dallas: 64.1%, Ny: 66.1%, La: 64.6%. These rates are close to those in
the main set, but evidence synthesis is conditional on accessible/extractable sources. Missing
metadata is zero; inaccessible pages remain represented as failed/incomplete retrievals.

## Files

- `pooled_aggregate_comparison.csv`: aggregate main-versus-robustness effects.
- `pooled_dimension_comparison.csv`: comparisons for each social dimension.
- `location_dimension_comparison.csv`: comparisons within each location and dimension.
- `comparison_summary.csv`: compact directional and uncertainty diagnostics.
- `pooled_effect_comparison.png`: pooled forest comparison across dimensions.
