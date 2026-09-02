# Data workspace

- `raw/`: immutable source snapshots.
- `interim/`: extracted, normalized, filtered, and deduplicated stages.
- `processed/`: versioned mixtures ready for tokenization.
- `tokenized/`: token shards before sequence packing.
- `packed/`: fixed-length training shards.
- `validation/`: held-out validation corpora.
- `manifests/`: committed provenance, hashes, counts, and token statistics.

Large data files are ignored by Git. Never modify `raw/` in place.

