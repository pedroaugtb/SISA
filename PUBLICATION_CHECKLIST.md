# Publication checklist

Complete these items before making the repository public:

- Confirm that peer-review policy permits public sharing; the current PDF explicitly prohibits it during review.
- Resolve `KNOWN_DISCREPANCY.md` and make the manuscript, rating files, and scale definition agree.
- Replace anonymous title-page metadata with authors, affiliations, acknowledgements, and funding/conflict statements as applicable.
- Add the final paper citation, DOI or preprint URL, and repository archival DOI.
- Choose and add a code license. No license has been selected automatically because this is an author/legal decision.
- Add explicit data-use terms for the collected Google AI Overview text and cited-source metadata if required by the venue or institution.
- Confirm that redistribution of generated AIO text complies with applicable platform terms and institutional guidance.
- Review the final source manifests for any additional fields the authors do not want public.
- Run `python3 run.py validate` and `python3 -m unittest discover -s tests -v` in a clean clone.
- Run `python3 run.py all-derived` and compare generated statistics with the manuscript.
- Create a tagged release and archive it with a long-term repository such as Zenodo or OSF.
