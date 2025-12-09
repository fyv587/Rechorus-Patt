# ReChorus-Patt: Probabilistic Attention for Sequential Recommendation

This repository is an implementation of **PAtt (Probabilistic Attention)** based on the [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) framework.

It reproduces the experiments from the paper: **"Probabilistic Attention for Sequential Recommendation" (KDD 2024)**.

## 🌟 Overview

This project extends ReChorus to include the **PAtt** model and provides scripts to reproduce the main experimental results, including:
1.  **Performance Comparison**: Comparing PAtt/DPAtt with state-of-the-art baselines.
2.  **Parameter Sensitivity**: Analyzing the impact of Dropout, Embedding Size, Learning Rate, Layers, and Heads as discussed in the paper (Section 4.1 & Appendix A.2).

## 🔧 Requirements & Installation

Please refer to the original [ReChorus Installation Guide](docs/Getting_Started.md) or simply install the dependencies:

```bash
pip install -r requirements.txt

📂 Data Preparation
Ensure your datasets are placed in the data/ directory. The structure should look like this:

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

没问题，为了方便你直接复制使用，我将完整的 README.md 内容整合在一个代码块中了。

你可以直接点击代码块右上角的 "Copy" 按钮，然后粘贴到你项目根目录的 README.md 文件里。

Markdown

# ReChorus-Patt: Probabilistic Attention for Sequential Recommendation

This repository is an implementation of **PAtt (Probabilistic Attention)** based on the [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) framework.

It reproduces the experiments from the paper: **"Probabilistic Attention for Sequential Recommendation" (KDD 2024)**.

## 🌟 Overview

This project extends ReChorus to include the **PAtt** model and provides scripts to reproduce the main experimental results, including:
1.  **Performance Comparison**: Comparing PAtt/DPAtt with state-of-the-art baselines.
2.  **Parameter Sensitivity**: Analyzing the impact of Dropout, Embedding Size, Learning Rate, Layers, and Heads as discussed in the paper (Section 4.1 & Appendix A.2).

## 🔧 Requirements & Installation

Please refer to the original [ReChorus Installation Guide](docs/Getting_Started.md) or simply install the dependencies:

```bash
pip install -r requirements.txt
📂 Data Preparation
Ensure your datasets are placed in the data/ directory. The structure should look like this:

Plaintext

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
🚀 Reproduction Scripts
We provide two primary shell scripts in the src/ directory to automate the experiments.

1. Baseline Comparison (src/run.sh)
Use this script to run the Main Performance Comparison (similar to Table 2 in the paper). It runs various baseline models on the target datasets.

Included Models:

Caser, FPMC, KDA, SLRCPlus, TiMiRec, TiSASRec.

Usage:

cd src
chmod +x run.sh
./run.sh

没问题，为了方便你直接复制使用，我将完整的 README.md 内容整合在一个代码块中了。

你可以直接点击代码块右上角的 "Copy" 按钮，然后粘贴到你项目根目录的 README.md 文件里。

Markdown

# ReChorus-Patt: Probabilistic Attention for Sequential Recommendation

This repository is an implementation of **PAtt (Probabilistic Attention)** based on the [ReChorus 2.0](https://github.com/THUwangcy/ReChorus) framework.

It reproduces the experiments from the paper: **"Probabilistic Attention for Sequential Recommendation" (KDD 2024)**.

## 🌟 Overview

This project extends ReChorus to include the **PAtt** model and provides scripts to reproduce the main experimental results, including:
1.  **Performance Comparison**: Comparing PAtt/DPAtt with state-of-the-art baselines.
2.  **Parameter Sensitivity**: Analyzing the impact of Dropout, Embedding Size, Learning Rate, Layers, and Heads as discussed in the paper (Section 4.1 & Appendix A.2).

## 🔧 Requirements & Installation

Please refer to the original [ReChorus Installation Guide](docs/Getting_Started.md) or simply install the dependencies:

```bash
pip install -r requirements.txt
📂 Data Preparation
Ensure your datasets are placed in the data/ directory. The structure should look like this:

Plaintext

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
🚀 Reproduction Scripts
We provide two primary shell scripts in the src/ directory to automate the experiments.

1. Baseline Comparison (src/run.sh)
Use this script to run the Main Performance Comparison (similar to Table 2 in the paper). It runs various baseline models on the target datasets.

Included Models:

Caser, FPMC, KDA, SLRCPlus, TiMiRec, TiSASRec.

Usage:

Bash

cd src
chmod +x run.sh
./run.sh
Configuration: You can modify the MODELS and DATASETS arrays in run.sh to select specific baselines or datasets.

Logs: Results will be saved to ../logs/<ModelName>/<Dataset>/train.log.

2. PAtt Parameter Sensitivity (src/run_parameters.sh)
Use this script to analyze the Hyper-parameter Sensitivity of the PAtt model (addressing RQ1 & RQ2). It performs a grid search over key parameters as described in the paper's Appendix A.2.

Explored Parameters:

Dropout: [0.3, 0.5, 0.7]

Embedding Size: [32, 64, 128]

Learning Rate: [1e-3, 1e-4]

Model Depth (Layers): [1, 2, 3]

Attention Heads: [1, 2, 4]

Usage:

Bash

cd src
chmod +x run_parameters.sh
./run_parameters.sh
Logs: Detailed logs for each parameter configuration will be saved to ../logs_hyper_general/PAtt/<Dataset>/.

📊 Evaluation Metrics
The framework evaluates models using the following metrics (Top-k = 5, 20):

NDCG (Normalized Discounted Cumulative Gain)

HR (Hit Rate / Recall)

📝 Citation
If you find this code useful, please cite the original ReChorus paper and the PAtt paper:

代码段

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
