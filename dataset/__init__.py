"""tinybrain dataset：预训练四档数据和后训练数据的完整流水线。

    sources.py   所有来源一张表：出自哪里、许可证、怎么下载、怎么取正文
    download.py  下载原始数据
    clean.py     单篇文档的清洗规则（含切书）
    pretrain.py  按来源批量清洗预训练数据
    mix.py       去重、配比、分四档写出、检查、生成数据集说明
    sft.py       SFT 语料
    rl.py        RL 题目、评测题、MOPD 提示
    build.py     总入口：tinygo data（--check 只检查每一步的输入）

数据都在 data/ 下，不进仓库：data/raw（原始下载）、data/clean（洗好的）、data/inputs（后训练原料）、data/*.parquet（成品）。
"""
