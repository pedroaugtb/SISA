# Evidence-alignment chunking sensitivity

This analysis compares the effective primary configuration (source 112/40; answer 96/24) with a smaller-window configuration (80/30; 64/20) and a same-encoder high-context/high-overlap configuration (112/56; 112/32).

The source window cannot defensibly exceed 112 under the primary encoder: the production pipeline caps requested windows at `model.max_seq_length - 16`. Calling a nominal 160-token rerun would therefore reproduce 112-token windows.

Eligibility is frozen from the primary analysis by location and response file. Effects are evidence alignment for focal minus comparison conditions. Location pairs are formed before repetitions are averaged within outcome. Absolute cosine levels are not treated as comparable across chunking configurations; conclusions use effect directions, ranks, and outcome-blocked associations.
