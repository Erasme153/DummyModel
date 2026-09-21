# Ladder v001

`p039m.yaml` now contains the complete M1 architecture (38,937,088 parameters).
`scripts/train/pretrain.py` reads its `model_config` directly, without recursive
YAML inheritance. Training settings stay in command-line arguments.

The larger target remains a placeholder. This directory does not yet represent
a validated scaling ladder.

Create `../v002/` instead of editing this ladder after comparable training runs
have begun.
