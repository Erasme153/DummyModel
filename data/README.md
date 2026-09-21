# Data

`tiny_corpus.txt` 是可提交 Git 的小型测试语料。M1 使用已下载到项目外的
`../data/fineweb-edu` 原始 Parquet，由 `scripts/data/prepare_fineweb.py` 抽样、
轻量过滤、精确去重、按文档划分、编码及 packing。

生成的数据放 `data/tokenized/m01_fineweb_100m/`，含独立的 `train.bin` 和
`validation.bin`、tokenizer 副本、文档索引、预览及一份自动生成的 `summary.json`。
二进制格式为小端 uint16，每 2048 个 ID 组成一行。只有摘要中 `status=complete`
才可进入训练，训练代码还应核对文件长度。

读取时先转换为 int64 再传给 PyTorch Embedding；向模型传入相同的 input_ids
和 labels，由模型内部完成 next-token shift。

命令、数据来源、处理规则及限制见
[`M1 实验说明`](../experiments/m01_pretraining/exp001_39m/README.md)。大型数据和
生成索引由 Git 忽略，不维护额外 manifest，也不复制几百 GB 原始语料到项目内。
