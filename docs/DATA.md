# Data and evidence availability

The reported runs use two fixed FineWeb-Edu sample-10BT shards and the
Qwen3-0.6B tokenizer. Full pinned dataset/tokenizer revisions and the principal
scientific settings are in `configs/study.json`.

The original primary and common follow-up evaluation each contain 1,024
sequences. Every score supervises 1,023 next-token targets per sequence.
The sets use distinct documents from the same preselected exclusion pool;
all selected document identities were excluded from the training streams.
Document-ID exclusion is not a semantic or near-duplicate audit. The manuscript
specifies the document-ranking salt, packing, EOS and masking conventions.
`configs/data-policies/` contains portability templates with personal local
paths removed and the original policy hashes recorded. These templates are
not byte-identical copies of the original policies and cannot substitute for
their hash-bound runtime identities. A new data build must bind its own paths
explicitly rather than pretending to replay the private deployment.

`results/scores/` contains curated numerical projections of the original score
records. It preserves loss, every sequence's summed NLL and target count,
seed, condition, evaluation and checkpoint identities. Infrastructure metadata
is omitted. The original record's SHA-256 is retained; a curated projection
does not claim byte identity with the private original record.

Model checkpoints, token payloads and tokenizer assets are not redistributed
in this repository. Checkpoint-manifest identifiers establish which saved
model was scored; they do not make the weights downloadable. Release terms
and hosting for those payloads remain unresolved. Historical cloud deployment
logs, credentials, rental accounting and private Git history are excluded.

The 33 endpoint records cover 24 models, including nine scored twice. They
are not 33 independent training runs. Original-primary losses and follow-up
losses must be analyzed separately.
