# Reproducing the paper

## Saved results

Run `python scripts/verify_results.py` from the repository root. Only Python's
standard library is needed. The script verifies the saved-result checksums,
aggregates per-sequence NLL sums and target counts, and reproduces all 33 scores,
paired training-seed means/SDs and original improvement decisions. It keeps
the two evaluation artifacts separate and checks the reuse of nine checkpoints.

## Model examples

Install `requirements-model.txt`, then run `python scripts/model_smoke.py`.
The example constructs all eight patterns, assigns nonzero depth-routing
queries, and checks finite forward/backward execution on reduced CPU FP32
models. It uses the real width-1,280 query/GQA layout but two layers, a small
vocabulary and a short synthetic sequence. This is a software check, not a
training result or CUDA validation.

```python
from depth_read_sharing import build_config, build_model, strict_runtime

with strict_runtime(fixture=True):
    config = build_config("AAAB", fixture=True)
    model = build_model("AAAB", fixture=True)
```

`fixture=False` selects the reported 36-layer configuration. Constructing that
model allocates approximately 1.08B parameters. The numerical settings are
listed in `configs/study.json`; the scientific runs used PyTorch 2.8.0+cu128,
Transformers 4.57.6 and native BF16 parameters and optimizer moments.

The retained model classes and read operators come from the evaluated source.
Historical internal class/filenames remain for correspondence. Packaging-only
changes and original source identities are recorded in
`reference/PROVENANCE.json`. Cloud controllers, preflight contracts and launch
authorization machinery are outside this paper repository.

## Training and data

The full training and data specifications are in `configs/study.json`,
`configs/data.json` and the manuscript. The repository does not include a
portable end-to-end trainer, model checkpoints or packed token datasets.
A complete rerun requires obtaining the pinned corpus/tokenizer and rebuilding
the document-excluded streams; the model examples do not perform that work.

The study uses batch 4, length 1,024, accumulation 8 and 2,000 updates of a
20,000-update schedule, including 1,000 warmup updates. Common initialization
and data order are matched within each seed. AABC and ABBC start from identical
three-read tensors and differ in assignment. The four additional two-read
conditions use a different driver from the reference models; the early parity
check does not establish full-horizon driver equivalence.

## Manuscript

Compile `paper/main.tex` with a modern LaTeX distribution. Its bibliography is
inline, so no BibTeX step is required. Figure PDFs and editable SVGs are in
`paper/figures/`; the included numerical tables are in `paper/tables/`.
