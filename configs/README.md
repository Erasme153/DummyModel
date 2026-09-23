# Configs

简单实验优先使用脚本默认值和少量命令行参数，不为每次运行创建 YAML。

只有参数需要复用、组合或批量 sweep 时才在这里增加配置。当前 M1 使用
`model/ladder/v001/p039m.yaml`，M2 使用 `model/p099m.yaml`；其余 model ladder
和 scaling 文件仍是草案。模型结构配置中不要写本机路径。
