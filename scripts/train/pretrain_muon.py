#!/usr/bin/env python3
"""Muon / AdamW 单卡或多卡预训练入口；复用 pretrain_ddp.py 的数据与恢复逻辑。

默认隐藏层二维权重使用 Muon，embedding、head、norm 使用 AdamW；
--optimizer adamw 可切回现有 DDP 基线。参数与更新范数按 --log-every 记录。
"""

from pathlib import Path
import sys

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train import pretrain_ddp  # noqa: E402


def main():
    parser = pretrain_ddp.build_parser(description=__doc__)
    parser.set_defaults(optimizer="muon", record_update_norms=True)
    pretrain_ddp.main(pretrain_ddp.parse_args(parser))


if __name__ == "__main__":
    main()
