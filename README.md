# ReChorus-Patt: 面向序列推荐的概率注意力机制 (Probabilistic Attention for Sequential Recommendation)

本项目基于 [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) 框架实现了 **PAtt (Probabilistic Attention)** 模型。

它复现了论文 **"Probabilistic Attention for Sequential Recommendation" (KDD 2024)** 中的实验。

## 🌟 项目概述 (Overview)

本项目扩展了 ReChorus 框架以包含 **PAtt** 模型，并提供了复现主要实验结果的脚本，包括：
1.  **性能对比 (Performance Comparison)**：将 PAtt/DPAtt 与最先进的基线模型进行对比。
2.  **参数敏感性 (Parameter Sensitivity)**：分析论文（第 4.1 节及附录 A.2）中讨论的 Dropout、嵌入维度 (Embedding Size)、学习率 (Learning Rate)、层数 (Layers) 和注意力头数 (Heads) 的影响。

## 🔧 环境要求与安装 (Requirements & Installation)

请参考原版 [ReChorus 安装指南](docs/Getting_Started.md) 或直接安装依赖项：

```bash
pip install -r requirements.txt
```

📂 数据准备 (Data Preparation)
请确保将数据集放置在 data/ 目录下。目录结构应如下所示：

````markdown
ReChorus-Patt/
├── data/
│   ├── MovieLens_1M/
│   ├── Grocery_and_Gourmet_Food/
│   └── MIND_Large/
├── src/
│   ├── main.py
│   ├── run.sh
│   └── run_parameters.sh
└── ...
````

🚀 复现脚本 (Reproduction Scripts)
我们在 src/ 目录下提供了两个主要的 Shell 脚本来自动化实验流程。

1. 基线模型对比 (src/run.sh)
使用此脚本运行 主要性能对比实验（类似于论文中的表 2）。它会在目标数据集上运行各种基线模型。

包含的模型：

Caser, FPMC, KDA, SLRCPlus, TiMiRec, TiSASRec.

使用方法：
```bash
cd src
chmod +x run.sh
./run.sh
```

配置： 你可以修改 run.sh 中的 MODELS 和 DATASETS 数组来选择特定的基线模型或数据集。

日志： 结果将保存在 ../logs/\<ModelName\>/\<Dataset\>/train.log 中。

2. PAtt 参数敏感性分析 (src/run_parameters.sh)
使用此脚本分析 PAtt 模型的 超参数敏感性（对应 RQ1 & RQ2）。它会对论文附录 A.2 中描述的关键参数进行网格搜索。

探索的参数：

Dropout (丢弃率): [0.3, 0.5, 0.7]

Embedding Size (嵌入维度): [32, 64, 128]

Learning Rate (学习率): [1e-3, 1e-4]

Model Depth (层数): [1, 2, 3]

Attention Heads (注意力头数): [1, 2, 4]

使用方法：
```bash
cd src
chmod +x run_parameters.sh
./run_parameters.sh
```

日志： 每个参数配置的详细日志将保存在 ../logs_hyper_general/PAtt/\<Dataset\>/ 中。

📊 评估指标 (Evaluation Metrics)
本框架使用以下指标评估模型 (Top-k = 5, 20)：

NDCG (归一化折损累计增益)

HR (命中率 / 召回率)

📝 引用 (Citation)
如果您觉得此代码有用，请引用原始 ReChorus 论文和 PAtt 论文：
```bibtex
@inproceedings{liu2024probabilistic,
  title={Probabilistic Attention for Sequential Recommendation},
  author={Liu, Yuli and Walder, Christian and Xie, Lexing and Liu, Yiqun},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1956--1967},
  year={2024}
}

@inproceedings{li2024rechorus2,
  title={ReChorus2. 0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}
```
