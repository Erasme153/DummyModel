"""miniLLaMA 的单设备参考实现。

本文件只描述模型的数学结构，不包含优化器、数据加载、FSDP2、张量并行或
KV Cache。把单卡模型和分布式基础设施分开，便于先验证数值正确性，再由
TorchTitan 对同一个模型施加并行策略。

整体数据流如下（B=batch，T=sequence length，D=hidden size，V=vocab size）：

    input_ids [B, T]
        -> Token Embedding [B, T, D]
        -> N x TransformerBlock [B, T, D]
        -> Final RMSNorm [B, T, D]
        -> LM Head [B, T, V]
        -> vocabulary logits

每个 TransformerBlock 使用 Pre-Norm 结构：

    x = x + GQA(RMSNorm(x))
    x = x + SwiGLU(RMSNorm(x))

注意力内部的 GQA、RoPE 和 causal mask 实现在 ``attention.py`` 中。
"""

from __future__ import annotations

from dataclasses import dataclass

import torch
import torch.nn.functional as F
from torch import nn

from .attention import GroupedQueryAttention
from .config import MiniLlamaConfig
from .mlp import SwiGLUMLP
from .norm import RMSNorm


@dataclass
class MiniLlamaOutput:
    """模型前向传播的返回值。

    Attributes:
        logits:
            每个位置对完整词表的未归一化分数，形状为 ``[B, T, V]``。
            这里没有执行 softmax；交叉熵内部会完成 log-softmax，推理端也可
            根据 temperature、top-k、top-p 等策略自行处理 logits。
        loss:
            传入 ``labels`` 时计算出的 next-token 平均交叉熵；没有传入 labels
            时为 ``None``。
    """

    logits: torch.Tensor
    loss: torch.Tensor | None = None


class TransformerBlock(nn.Module):
    """一个 Pre-Norm Decoder Transformer Block。

    结构严格对应：

        RMSNorm -> GQA Causal Attention(+RoPE) -> Residual
        RMSNorm -> SwiGLU MLP                  -> Residual

    Pre-Norm 指的是先归一化再进入子层。残差支路上的原始 hidden states 不做
    归一化，可以为深层网络提供更直接的梯度通路。
    """

    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()

        # 注意力子层之前的 RMSNorm。它只有一个形状为 [D] 的可训练缩放参数，
        # 不减均值，也没有 bias；具体计算见 norm.py。
        self.input_layernorm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # Grouped-Query Self-Attention：Q head 数量可以多于 K/V head 数量。
        # attention.py 会在 Q/K 上应用 RoPE，并通过 PyTorch SDPA 实现 causal
        # attention。该模块的输入和输出形状均为 [B, T, D]。
        self.self_attn = GroupedQueryAttention(config)

        # MLP 子层之前的第二个 RMSNorm。
        self.post_attention_layernorm = RMSNorm(
            config.hidden_size, config.rms_norm_eps
        )

        # SwiGLU 前馈网络：silu(gate_proj(x)) * up_proj(x)，再经 down_proj
        # 投影回 hidden size，因此其输入输出也都是 [B, T, D]。
        self.mlp = SwiGLUMLP(config)

    def forward(
        self,
        hidden_states: torch.Tensor,
        position_ids: torch.Tensor,
        attention_mask: torch.Tensor | None = None,
    ) -> torch.Tensor:
        """执行一个 Transformer Block。

        Args:
            hidden_states: 上一层输出，形状 ``[B, T, D]``。
            position_ids: 每个 token 的位置编号，形状 ``[B, T]``，供 RoPE 使用。
            attention_mask: 可选 padding mask，形状 ``[B, T]``；1/True 表示有效
                token，0/False 表示不能作为 key 被关注的 padding token。

        Returns:
            与输入形状相同的 ``[B, T, D]`` hidden states。
        """

        # 第一条残差：保存未经归一化的输入 x。
        residual = hidden_states

        # Pre-Norm attention：Attn(RMSNorm(x))。
        hidden_states = self.input_layernorm(hidden_states)
        hidden_states = self.self_attn(
            hidden_states,
            position_ids=position_ids,
            attention_mask=attention_mask,
        )

        # 残差相加：x <- x + Attention(RMSNorm(x))。
        hidden_states = residual + hidden_states

        # 第二条残差从 attention 子层的完整输出开始保存。
        residual = hidden_states

        # Pre-Norm MLP：MLP(RMSNorm(x))。
        hidden_states = self.post_attention_layernorm(hidden_states)
        hidden_states = self.mlp(hidden_states)

        # 残差相加：x <- x + MLP(RMSNorm(x))。
        return residual + hidden_states


class MiniLlamaForCausalLM(nn.Module):
    """用于 next-token prediction 的完整 miniLLaMA 模型。

    这是一个 decoder-only causal language model。第 t 个位置产生的 logits
    只能依赖第 0..t 个输入 token，并用于预测第 t+1 个 token。
    """

    def __init__(self, config: MiniLlamaConfig) -> None:
        super().__init__()

        # 保存架构配置，供初始化、forward、参数统计和未来 checkpoint 导出使用。
        self.config = config

        # 将离散 token ID 映射为连续向量：
        #   输入  input_ids:    [B, T]
        #   权重  embedding:    [V, D]
        #   输出  hidden_states:[B, T, D]
        self.embed_tokens = nn.Embedding(
            config.vocab_size,
            config.hidden_size,
            padding_idx=config.pad_token_id,
        )

        # ModuleList 会把每个 TransformerBlock 注册为 PyTorch 子模块，使其参数
        # 能被 parameters()、state_dict()、to(device) 和 FSDP 等机制发现。
        self.layers = nn.ModuleList(
            TransformerBlock(config) for _ in range(config.num_hidden_layers)
        )

        # 所有 decoder block 之后再做一次最终 RMSNorm，这是 LLaMA 类模型的
        # 常见结构。输出形状仍为 [B, T, D]。
        self.norm = RMSNorm(config.hidden_size, config.rms_norm_eps)

        # 将每个位置的 hidden vector 投影到词表空间：D -> V。
        # bias=False 与 LLaMA-like 设计一致。
        self.lm_head = nn.Linear(config.hidden_size, config.vocab_size, bias=False)

        # nn.Module.apply 会递归访问本模型、所有子模块以及更深层子模块，并对
        # 每个模块调用 _initialize_module。只有 Linear 和 Embedding 会命中下面
        # 的类型判断；RMSNorm 保持其构造时的全 1 权重，RoPE buffer 也不改变。
        self.apply(self._initialize_module)

        # 可选的 weight tying：让输出层和输入 embedding 引用同一个 Parameter。
        # 这样不仅减少 V*D 个参数，也使输入/输出 token 表示共享同一套权重。
        # 此赋值发生在初始化之后，所以最终保留的是 embed_tokens.weight。
        if config.tie_word_embeddings:
            self.lm_head.weight = self.embed_tokens.weight

    def _initialize_module(self, module: nn.Module) -> None:
        """初始化 ``self.apply`` 遍历到的一个子模块。

        ``self.apply(self._initialize_module)`` 会递归调用本函数。这里只处理
        Linear 和 Embedding，统一从 N(0, initializer_range^2) 初始化。

        Args:
            module: 当前递归访问到的某个 ``nn.Module``。它可能是 Linear、
                Embedding、RMSNorm、TransformerBlock、ModuleList 或整个模型。
        """

        if isinstance(module, nn.Linear):
            # normal_ 末尾的下划线表示原地写入 module.weight。
            # torch.nn.init 中的初始化函数内部已禁用 autograd，因此这里不需要
            # 额外包一层 torch.no_grad()。
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
        elif isinstance(module, nn.Embedding):
            # Token embedding 使用与 Linear 相同的初始化分布，方便不同规模模型
            # 保持一致的初始化约定。
            nn.init.normal_(
                module.weight,
                mean=0.0,
                std=self.config.initializer_range,
            )
            if module.padding_idx is not None:
                # 上面的随机初始化覆盖了 padding 行，因此将它恢复为全 0。
                # 这是对 requires_grad=True 的叶子 Parameter 做原地修改，所以
                # 显式关闭梯度记录。nn.Embedding 也会阻止 padding 行累积梯度。
                with torch.no_grad():
                    module.weight[module.padding_idx].zero_()

    def num_parameters(self, *, trainable_only: bool = False) -> int:
        """返回模型参数总数，而不是参数 Tensor 的数量。

        Args:
            trainable_only: 为 True 时只统计 ``requires_grad=True`` 的参数。

        PyTorch 的 parameters() 会对共享 Parameter 去重，因此启用 embedding/
        LM-head weight tying 后，共享矩阵只统计一次。
        """

        parameters = self.parameters()
        if trainable_only:
            parameters = (
                parameter for parameter in parameters if parameter.requires_grad
            )

        # numel() 返回每个 Parameter 包含的标量个数。
        return sum(parameter.numel() for parameter in parameters)

    def forward(
        self,
        input_ids: torch.Tensor,
        *,
        attention_mask: torch.Tensor | None = None,
        position_ids: torch.Tensor | None = None,
        labels: torch.Tensor | None = None,
    ) -> MiniLlamaOutput:
        """完成前向传播，并可选地计算 causal language-modeling loss。

        Args:
            input_ids: token ID，形状 ``[B, T]``，通常 dtype 为 torch.long。
            attention_mask: 可选 padding mask，形状 ``[B, T]``。不传时，SDPA
                直接使用高效的 ``is_causal=True`` 路径。
            position_ids: 可选位置编号，形状 ``[B, T]``。不传时自动生成
                ``0, 1, ..., T-1``。自定义它可支持 padding 或后续 KV cache。
            labels: 可选训练目标，形状 ``[B, T]``。通常令 labels=input_ids；
                其中值为 -100 的位置会被 loss 忽略。

        Returns:
            ``MiniLlamaOutput(logits=[B, T, V], loss=scalar_or_None)``。
        """

        # Embedding 期望二维 token ID。提前检查能给出比底层算子更清晰的报错。
        if input_ids.ndim != 2:
            raise ValueError("input_ids must have shape [batch_size, sequence_length]")
        batch_size, sequence_length = input_ids.shape

        # 当前 RoPE 配置只声明支持到 max_position_embeddings；超长输入应由数据
        # packing 阶段截断或切块，而不是在模型内部静默处理。
        if sequence_length > self.config.max_position_embeddings:
            raise ValueError(
                f"sequence length {sequence_length} exceeds "
                f"max_position_embeddings={self.config.max_position_embeddings}"
            )

        if position_ids is None:
            # arange 先得到 [T]，unsqueeze 后为 [1, T]，expand 得到 [B, T]。
            # expand 通常只创建广播视图，不为每个 batch 复制一份位置数组。
            position_ids = torch.arange(
                sequence_length,
                device=input_ids.device,
                dtype=torch.long,
            ).unsqueeze(0).expand(batch_size, -1)
        elif position_ids.shape != input_ids.shape:
            raise ValueError("position_ids must have the same shape as input_ids")

        # [B, T] -> [B, T, D]
        hidden_states = self.embed_tokens(input_ids)

        # 依次通过 N 个 decoder block；每层都保持 [B, T, D] 形状。
        for decoder_layer in self.layers:
            hidden_states = decoder_layer(
                hidden_states,
                position_ids=position_ids,
                attention_mask=attention_mask,
            )

        # 最终归一化，然后把 hidden size 投影到 vocabulary size：
        # [B, T, D] -> [B, T, V]。
        hidden_states = self.norm(hidden_states)
        logits = self.lm_head(hidden_states)

        # 推理时 labels=None，只返回 logits；训练时才计算 loss。
        loss = None
        if labels is not None:
            if labels.shape != input_ids.shape:
                raise ValueError("labels must have the same shape as input_ids")
            if sequence_length < 2:
                raise ValueError("at least two tokens are required to compute causal loss")

            # Next-token prediction 需要错开一位：
            #
            #   input/labels:  [t0, t1, t2, t3]
            #   使用的 logits: [L0, L1, L2]      （去掉最后一个位置）
            #   预测的 labels: [t1, t2, t3]      （去掉第一个位置）
            #
            # 因此 L0 学习预测 t1，L1 学习预测 t2，L2 学习预测 t3。
            # 最后一个 logits 没有序列内的“下一个 token”目标，所以不参与 loss。
            # 转成 float32 可以提高 bf16/fp16 训练时交叉熵的数值稳定性。
            shift_logits = logits[:, :-1, :].contiguous().float()
            shift_labels = labels[:, 1:].contiguous()

            # CrossEntropy 接收 [样本数, 类别数]，所以把 batch 和 sequence 两个
            # 维度展平：[B, T-1, V] -> [B*(T-1), V]。
            # contiguous() 确保切片后的内存布局可以安全地使用 view()。
            loss = F.cross_entropy(
                shift_logits.view(-1, self.config.vocab_size),
                shift_labels.view(-1),
                # 数据 collator 可将 padding、跨文档边界或不希望监督的位置标为
                # -100；这些位置不会进入 loss 的平均值。
                ignore_index=-100,
            )

        return MiniLlamaOutput(logits=logits, loss=loss)
