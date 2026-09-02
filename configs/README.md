# Configuration layout

- `model/ladder/vNNN/`: one immutable set of model-size configurations.
- `scaling/`: ladder manifests and scaling-law experiment definitions.
- `data/` and `tokenizer/`: versioned corpus and tokenizer recipes.
- `training/`: hardware/runtime settings, separate from model architecture.
- `evaluation/`, `posttrain/`, `inference/`, `profiling/`: downstream recipes.

Do not encode GPU count or local paths in model architecture files.

