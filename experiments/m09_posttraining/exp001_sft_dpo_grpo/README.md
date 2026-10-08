# M9：SFT、DPO 与 GSM8K GRPO

## 结论

M9 已完成 SFT、DPO 和可验证奖励 GRPO 的训练与评测。UltraFeedback SFT
的独立回答 loss 从 **2.921682** 降至 **2.060377**；DPO 在另一份
844 对偏好验证集上的 loss 从 **0.693147** 降至 **0.614637**，相对
reference 的奖励排序正确率达到 **67.06%**。数学分支先经 GSM8K SFT
建立非零奖励信号，再做 GRPO；官方 1319 题测试的 pass@4 从
**62/1319（4.70%）**升至 **85/1319（6.44%）**，但首次采样正确仅
**18 → 25 题**。GRPO 输出高度集中于少数常见数字，现有结果不能证明
数学推理能力实质提高。M9 不追加超参数搜索。

## 设置与结果

四个模型均为 213M 参数、M4 的 Mistral 32K tokenizer、双卡 BF16 DDP、
AdamW、seed=2026、最大长度 2048。UltraFeedback SFT 从 M8 纯 FineWeb
cooldown checkpoint 初始化；DPO 和 GSM8K SFT 分别从该 SFT checkpoint
分支。GRPO 从 GSM8K SFT 初始化，以同一个 checkpoint 为冻结 reference；
因此 DPO 与 GRPO 不是同一数学任务上的直接对照。

UltraFeedback 采用 `[INST] prompt [/INST]` 模板，只监督 assistant 回答
和 EOS。审计后保留 **60,863** 条 SFT 训练、**996** 条 SFT 验证；
DPO 过滤同分/逆序、相同回答、重复或跨划分 prompt、空回答及超长配对后，
保留 **53,061** 训练对、**844** 独立验证对。GSM8K 训练原始 7473 题，
排除与 UltraFeedback 重叠的 3 题，按 seed=2026 固定留出 256 题，
剩余 **7214** 题训练；官方 **1319** 题测试答案未参与训练或选模型。

| 阶段（`runs/` 目录） | 主要配方 | 独立验证结果 |
| --- | --- | --- |
| `m09_213m_sft` | 全局 batch 16，LR `2e-5`，1 epoch / 3803 步 | UltraFeedback 回答 loss **2.921682 → 2.060377** |
| `m09_213m_dpo` | 全局 batch 8，LR `5e-6`，β=0.1，1 epoch / 6632 步 | 偏好 loss **0.693147 → 0.614637**；奖励排序正确率 **67.06%** |
| `m09_213m_gsm8k_sft` | 全局 batch 16，LR `2e-5`，1 epoch / 450 步 | GSM8K 回答 loss **2.366634 → 0.902941** |
| `m09_213m_grpo_g8` | 每题 8 个采样、每组 2 次更新、LR `5e-6`、KL 系数 0.01；901 轮 / 1802 次更新 | 256 题留出集：pass@1 **5/256 → 5/256**；pass@4 **10/256 → 20/256** |

DPO 验证的相对 reference 奖励间隔为 **0.61335**；直接比较 policy
对 chosen/rejected 的序列 log-prob，正确率仅从 **40.88%** 到
**41.59%**。这两个指标定义不同，不能把 67.06% 当作生成回答的
偏好胜率。GRPO 训练见 **7208** 题（6 题因全局 batch 尾部丢弃），
共 **57,664** 个采样，正确 **1181** 个；仅 **785/7208（10.89%）**
题出现组内正负奖励混合，奖励信号稀疏。

官方测试对 GSM8K SFT 与 GRPO 使用同一 1319 题、每题 4 次采样、
最多 256 新 token、temperature=0.7、top-k=50、top-p=0.95；这是
训练结束后的最终评测，不再用于调参。

| 官方测试指标 | GSM8K SFT | GRPO |
| --- | ---: | ---: |
| 可解析最终答案 | 4685/5276（88.80%） | 5232/5276（99.17%） |
| 首次采样正确 / pass@1 | 18/1319（1.36%） | 25/1319（1.90%） |
| 至少一次正确 / pass@4 | 62/1319（4.70%） | 85/1319（6.44%） |
| 平均生成 tokens | 132.1 | 91.7 |

成对题目比较，GRPO 的 pass@4 新增命中 72 题，同时失去 SFT 原本
命中的 49 题。GRPO 可解析答案仅有 **202** 种（SFT 为 **584** 种），
前 10 种占 **65.35%**（SFT 为 **29.75%**）。固定模型预测、随机
置换题目标准答案时，GRPO 的 pass@4 期望约 **89.46** 题，实际为
**85** 题；该诊断提示常见数字猜测可解释大部分表面增益，不能视为
严格的零推理能力证明。样例中仍可见错误算式。

同一 M4 通用验证集的 **9,993,454** 个预测位置上，UltraFeedback SFT、
DPO、GSM8K SFT、GRPO 的 loss 依次为 **3.018103、3.056975、
3.068799、3.073452**。DPO 比其 SFT 起点高 **0.038871**，数学 SFT
高 **0.050695**，GRPO 比数学 SFT 再高 **0.004653**；这说明通用文本
loss 有回退，但不能把不同任务的专用验证 loss 横向比较。

当前 GRPO 入口用温度及 top-k/top-p 截断分布采样，却从未截断的
logits 计算 old/current/reference log-prob。概率比因此不严格对应行为
采样分布；本次是教学实现与有限预算结果，不能用来断言标准 on-policy
GRPO 的效果。

## 复现与产物

在项目根目录执行。已有 `runs/` 输出目录不能覆盖；实际重跑时更换
`--output-dir`。以下显式列出影响本报告的关键训练参数，未列出的参数
沿用入口默认值。

```bash
CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/sft_ddp.py --dataset ultrafeedback \
  --raw-data-dir data/raw/m09_ultrafeedback/data \
  --init-checkpoint runs/m08_213m_cooldown/checkpoint.pt \
  --batch-size 4 --grad-accum-steps 2 --epochs 1 --learning-rate 2e-5 \
  --seed 2026 --output-dir runs/m09_213m_sft

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/dpo_ddp.py --raw-data-dir data/raw/m09_ultrafeedback/data \
  --init-checkpoint runs/m09_213m_sft/checkpoint.pt \
  --batch-size 1 --grad-accum-steps 4 --epochs 1 --learning-rate 5e-6 \
  --beta 0.1 --seed 2026 --output-dir runs/m09_213m_dpo

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/sft_ddp.py --dataset gsm8k \
  --raw-data-dir data/raw/m09_gsm8k/main \
  --ultrafeedback-dir data/raw/m09_ultrafeedback/data \
  --init-checkpoint runs/m09_213m_sft/checkpoint.pt \
  --holdout-size 256 --split-max-new-tokens 256 \
  --batch-size 4 --grad-accum-steps 2 --epochs 1 --learning-rate 2e-5 \
  --seed 2026 --output-dir runs/m09_213m_gsm8k_sft

CUDA_VISIBLE_DEVICES=0,1 PYTHONNOUSERSITE=1 torchrun --standalone --nproc-per-node=2 \
  scripts/train/grpo_ddp.py --data-dir data/raw/m09_gsm8k/main \
  --ultrafeedback-dir data/raw/m09_ultrafeedback/data \
  --init-checkpoint runs/m09_213m_gsm8k_sft/checkpoint.pt \
  --holdout-size 256 --max-new-tokens 256 --group-size 8 \
  --batch-size 1 --grad-accum-steps 4 --policy-epochs 2 --epochs 1 \
  --temperature 0.7 --top-k 50 --top-p 0.95 \
  --learning-rate 5e-6 --kl-coef 0.01 --clip-eps 0.2 \
  --seed 2026 --output-dir runs/m09_213m_grpo_g8

for run_name in m09_213m_gsm8k_sft m09_213m_grpo_g8; do
  CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 python scripts/eval/grpo_baseline.py \
    --data-dir data/raw/m09_gsm8k/main \
    --ultrafeedback-dir data/raw/m09_ultrafeedback/data \
    --checkpoint "runs/$run_name/checkpoint.pt" --split test \
    --holdout-size 256 --group-size 4 --max-new-tokens 256 \
    --temperature 0.7 --top-k 50 --top-p 0.95 --seed 2026 \
    --device cuda:0 --precision bf16 \
    --output "runs/m09_eval/${run_name}_test.json"
done
```

各训练目录有 `summary.json`、`checkpoint.pt` 与 TensorBoard 日志；
`runs/m09_eval/` 保存 GSM8K 留出/官方测试与 M4 通用验证 loss JSON。

## 后续

按 README 的 M9 范围，训练与最终评测到此结束，不再用已看过的
GSM8K 官方测试选择模型。若另立目标研究更严格的 GRPO 算法，应先使
采样分布与概率比定义一致，再用未参与当前选择的新评测数据做一次受控
比较；这属于新的研究问题，不是 M9 必需补跑项。
