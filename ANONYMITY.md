# Double-anonymous review safeguards

During peer review, share only the configured Anonymous GitHub URL. Keep the source GitHub repository private, do not cite or link it in the manuscript, and do not publish author metadata until the review policy permits disclosure.

This review artifact intentionally contains no paper-author names, affiliations, email addresses, ORCID identifiers, personal repository URLs, local filesystem paths, or manuscript acknowledgements. The Git history used for the source repository should likewise use a neutral author identity.

The release was audited across file and directory names, all text-based content, Git metadata, and PDF/PNG metadata. The anonymous service's public file listing and rendered README were also checked for source-owner information. Names and contact details that occur in cited third-party web material are not contributor metadata; one unnecessary third-party contact email embedded in an archived evidence excerpt was removed as an additional precaution.

Anonymous GitHub anonymizes text files but serves binary files as-is. For that reason, the included PDFs have no author, creator, producer, creation-date, modification-date, or XMP metadata. If figures are regenerated, audit their metadata again before pushing them.

No technical process can rule out deliberate authorship inference from writing style, topic familiarity, or information available elsewhere. This package is designed to prevent direct and metadata-based disclosure, while the private source repository prevents ordinary public search from linking the artifact to a GitHub account.
