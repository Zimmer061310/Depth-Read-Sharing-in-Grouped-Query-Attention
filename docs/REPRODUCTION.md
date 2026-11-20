# Reproduction and verification

## Saved numerical results

Run `python scripts/verify_results.py` from the repository root. This requires
only Python's standard library. The check independently aggregates all
per-sequence NLL sums and target counts, reproduces paired training-seed
means and sample SDs, preserves artifact separation and checks failed/successful
practical criteria. A release manifest verifies the distributed files.

## Model construction

Install `requirements-model.txt` for the model API. The scientific CUDA stack
used PyTorch 2.8.0+cu128, Transformers 4.57.6 and native BF16 parameters and
optimizer moments. `python scripts/model_smoke.py` uses reduced CPU FP32
fixtures with the real 1,280-wide query/GQA layout; it performs no optimizer
updates or held-out scoring. These are engineering checks, not replications.

```python
from depth_read_sharing import build_config, build_model
config = build_config("AAAB", fixture=True)
model = build_model("AAAB", fixture=True)
```

The production configuration is available with `fixture=False`. Constructing
a production model allocates approximately 1.08B parameters. Configure the
strict runtime explicitly before executing it; the CPU fixture command shows
the context-manager interface without changing scientific recipes.

The original three candidate classes and later shared-read/two-read controls
retain their distinct validators. No universal replacement model or silently
modified historical implementation is introduced. The read operator, physical
query ordering and native loss are defined in the manuscript. The exact
reference-source file hashes and originating commit are in `SOURCE-MAP.json`.

## Training and evaluation scope

The included reference modules provide the original training/data/checkpoint
primitives. The historical cloud controllers and paid execution machinery are
not a portable training command. A full rerun requires obtaining the upstream
corpus/tokenizer, constructing the frozen excluded-document streams, providing
independent storage and running the 24 specified trainings under the recorded
recipe. This release does not claim one-command end-to-end replication or
include the missing token/checkpoint payloads.

The study uses batch 4, length 1,024, accumulation 8 and exactly 2,000 updates
of the original 20,000-update schedule with 1,000 warmup updates. Preserve
common initialization and data order within each seed. AABC and ABBC start
from identical three-read tensors and differ in assignment. Additional seeds,
longer horizons, different operators or new evaluation sets are separate
studies. Short driver/runtime checks do not establish endpoint driver parity.

## Paper

`paper/main.tex` uses an inline bibliography and project-relative figure/table
files. Compile it with a modern LaTeX distribution; no BibTeX step is required.
Editable SVG figures are supplied alongside the PDFs. They are the existing
scientific figures, rather than regenerated measurements.

After deliberate documentation edits, use `python scripts/update_manifest.py`
to refresh existing release-file checksums before running verification. This
refuses changes to byte-preserved source files; changing a scientific source
requires an explicit new source mapping and validation. New files must also
be added explicitly to the release inventory.
