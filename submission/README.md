# arXiv v1 preparation

Build the upload and copyable form fields from the current checkout:

~~~sh
python scripts/verify_results.py
python scripts/prepare_arxiv.py
~~~

The generated files are under output/arxiv-v1/:

- arxiv-v1-source.zip: only the manuscript and its actual TeX dependencies.
- arxiv-form.txt and arxiv-form.json: title, author, expanded abstract,
  proposed categories and the author's selected manuscript license.
- BUILD-REPORT.json and SHA256SUMS: source inventory and upload checksums.

Upload only the source ZIP to arXiv. The bibliography is inline. Unused
figures, edit previews, source/evidence ledgers, operational records and
generated build products are excluded. The separate metadata/report files
are preparation records, not manuscript attachments.

The author selected the arXiv perpetual, non-exclusive license for the
manuscript and confirmed self-funding and no competing interests. This
selection does not change the model code's existing MIT license, release
model weights or token datasets, or change repository visibility.

Use the author-selected cs.CL primary category, with cs.LG as a proposed
cross-list. Classification is subject to arXiv moderation.
Check the title, author and abstract in the form, inspect the PDF produced
by arXiv, and verify the title page, Figure 1, source-depth maps, equations,
per-seed tables and page transitions before the author's final submission.
Local compilation and source ZIP verification do not replace that review.

No arXiv identifier is assigned during preparation. Update citation metadata
only after an identifier actually exists.

Official instructions checked on 21 November 2026:
[TeX uploads](https://info.arxiv.org/help/submit_tex.html),
[metadata](https://info.arxiv.org/help/prep.html),
[licenses](https://info.arxiv.org/help/license/index.html).
