# Data and evidence availability

The reported runs use two fixed FineWeb-Edu sample-10BT shards and the
Qwen3-0.6B tokenizer. Full pinned dataset/tokenizer revisions and the principal
scientific settings are in `configs/study.json` and `configs/data.json`.

The original primary and common follow-up evaluation each contain 1,024
sequences. Every score supervises 1,023 next-token targets per sequence.
The sets use distinct documents from the same preselected exclusion pool;
all selected document identities were excluded from the training streams.
Document-ID exclusion is not a semantic or near-duplicate audit. The manuscript
specifies the document-ranking salt, packing, EOS and masking conventions.
`configs/data.json` records the native document-identity rule, deterministic
ranking, exclusion-pool size, ordered source shards and tokenizer identities.
It describes the scientific data construction without local deployment paths.
The 4,096 held-out document identities are excluded before training shuffle,
tokenization and packing. Document streams append one EOS per document and
use windows of 1,025 stored positions with stride 1,024. The final input logit
is not supervised, giving 1,023 targets per example. EOS and document boundaries
are not masked out of the loss.

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
