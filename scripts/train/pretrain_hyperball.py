#!/usr/bin/env python3
"""DDP Hyperball 预训练：MuonH 或 AdamH 更新 Transformer 二维权重。

每个受约束矩阵固定初始化 Frobenius 范数；embedding/head/norm 用 AdamW。
--hyperball-lr 是相对更新长度，不能沿用 MuonW/AdamW 的学习率。
--weight-decay 只作用于辅助 AdamW 组，受约束矩阵不使用 weight decay。
"""

from pathlib import Path
import math
import sys

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from scripts.train import pretrain_ddp  # noqa: E402


class HyperballWithAuxAdamW(torch.optim.Optimizer):
    """以 Muon/Adam 产生方向，逐矩阵归一化更新并投影回初始半径。"""

    def __init__(self, model, args):
        hidden, aux_decay, aux_no_decay = pretrain_ddp.split_hidden_and_aux_parameters(model)
        adamw_lr = args.adamw_lr if args.adamw_lr is not None else args.learning_rate
        groups = [
            {"params": hidden, "norm_kind": args.optimizer, "lr": args.hyperball_lr,
             "use_muon": args.optimizer == "muonh", "betas": (args.beta1, args.beta2), "eps": 1e-8,
             "momentum": args.muon_momentum, "ns_steps": args.muon_ns_steps,
             "weight_decay": 0.0},
            {"params": aux_decay, "norm_kind": "adamw", "lr": adamw_lr,
             "use_muon": False, "betas": (args.beta1, args.beta2), "eps": 1e-8,
             "weight_decay": args.weight_decay},
            {"params": aux_no_decay, "norm_kind": "adamw", "lr": adamw_lr,
             "use_muon": False, "betas": (args.beta1, args.beta2), "eps": 1e-8,
             "weight_decay": 0.0},
        ]
        super().__init__(groups, defaults={})
        for parameter in hidden:
            radius = float(parameter.detach().norm())
            if not math.isfinite(radius) or radius <= 0:
                raise ValueError("Hyperball 需要每个受约束矩阵的初始范数为有限正数")
            self.state[parameter]["hyperball_radius"] = radius

    @staticmethod
    def _adam_statistics(grad, state, group):
        if "exp_avg" not in state:
            state["exp_avg"] = torch.zeros_like(grad)
            state["exp_avg_sq"] = torch.zeros_like(grad)
            state["step"] = 0
        state["step"] += 1
        beta1, beta2 = group["betas"]
        state["exp_avg"].lerp_(grad, 1 - beta1)
        state["exp_avg_sq"].mul_(beta2).addcmul_(grad, grad, value=1 - beta2)
        denominator = state["exp_avg_sq"].sqrt() / math.sqrt(1 - beta2 ** state["step"])
        denominator.add_(group["eps"])
        return state["exp_avg"], denominator, 1 - beta1 ** state["step"]

    @staticmethod
    @torch.no_grad()
    def _projected_step(parameter, direction, radius, learning_rate):
        # R * Normalize(W - eta * R * Normalize(u)); 零方向保留原矩阵。
        direction.div_(direction.norm().clamp_min(1e-12))
        parameter.add_(direction, alpha=-learning_rate * radius)
        parameter.mul_(radius / parameter.norm().clamp_min(1e-12))

    @torch.no_grad()
    def step(self, closure=None):
        loss = None
        if closure is not None:
            with torch.enable_grad():
                loss = closure()
        for group_index, group in enumerate(self.param_groups):
            for parameter in group["params"]:
                grad = parameter.grad
                if grad is None:
                    continue
                state = self.state[parameter]
                if group_index == 0:
                    if group["use_muon"]:
                        if "momentum_buffer" not in state:
                            state["momentum_buffer"] = torch.zeros_like(parameter)
                        momentum = state["momentum_buffer"]
                        momentum.lerp_(grad, 1 - group["momentum"])
                        direction = grad.lerp(momentum, group["momentum"])
                        update = pretrain_ddp._orthogonalize_muon(direction, group["ns_steps"])
                    else:
                        exp_avg, denominator, _ = self._adam_statistics(grad, state, group)
                        # Adam 的一阶偏差校正是正标量，归一化方向后会抵消。
                        update = exp_avg / denominator
                    self._projected_step(parameter, update, state["hyperball_radius"], group["lr"])
                else:
                    exp_avg, denominator, bias_correction = self._adam_statistics(grad, state, group)
                    parameter.mul_(1 - group["lr"] * group["weight_decay"])
                    parameter.addcdiv_(exp_avg, denominator,
                                        value=-group["lr"] / bias_correction)
        return loss


def main():
    parser = pretrain_ddp.build_parser(description=__doc__)
    parser.set_defaults(optimizer="muonh", record_update_norms=True)
    pretrain_ddp.main(pretrain_ddp.parse_args(parser))


if __name__ == "__main__":
    main()
