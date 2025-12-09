ReChorus-Patt: Probabilistic Attention for Sequential Recommendation

This repository is an implementation of PAtt (Probabilistic Attention) based on the ReChorus 2.0￼ framework.

It reproduces the experiments from the paper:

“Probabilistic Attention for Sequential Recommendation” (KDD 2024)

⸻

🌟 Overview

This project extends ReChorus to include the PAtt model and provides scripts to reproduce the primary experimental results:
	1.	Performance Comparison
Compare PAtt / DPAtt with state-of-the-art baselines.
	2.	Parameter Sensitivity
Analyze Dropout, Embedding Size, Learning Rate, Layers, and Heads
(Section 4.1 & Appendix A.2 of the paper).

⸻

🔧 Requirements & Installation

Refer to the original ReChorus guide:
docs/Getting_Started.md

Install dependencies:

pip install -r requirements.txt


⸻

📂 Data Preparation

Place your datasets inside the data/ directory. Expected structure:

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


⸻

🚀 Reproduction Scripts

Two main scripts inside src/ can reproduce the key results.

⸻

1. Baseline Comparison — src/run.sh

Runs experiments similar to Table 2 in the paper.

Built-in Models:
	•	Caser
	•	FPMC
	•	KDA
	•	SLRCPlus
	•	TiMiRec
	•	TiSASRec

Usage:

cd src
chmod +x run.sh
./run.sh

Modify models/datasets:
Edit MODELS and DATASETS in the script.

Logs saved at:

../logs/<ModelName>/<Dataset>/train.log


⸻

2. PAtt Parameter Sensitivity — src/run_parameters.sh

Runs the hyper-parameter sensitivity study.

Parameters:
	•	Dropout: 0.3, 0.5, 0.7
	•	Embedding Size: 32, 64, 128
	•	Learning Rate: 1e-3, 1e-4
	•	Layers: 1, 2, 3
	•	Heads: 1, 2, 4

Usage:

cd src
chmod +x run_parameters.sh
./run_parameters.sh

Logs saved at:

../logs_hyper_general/PAtt/<Dataset>/


⸻

📊 Evaluation Metrics

Top-K = 5, 20
	•	NDCG (Normalized Discounted Cumulative Gain)
	•	HR (Hit Rate)

⸻

📝 Citation

If you find this code useful, please cite:

@inproceedings{liu2024probabilistic,
  title={Probabilistic Attention for Sequential Recommendation},
  author={Liu, Yuli and Walder, Christian and Xie, Lexing and Liu, Yiqun},
  booktitle={Proceedings of the 30th ACM SIGKDD Conference on Knowledge Discovery and Data Mining},
  pages={1956--1967},
  year={2024}
}

@inproceedings{li2024rechorus2,
  title={ReChorus2.0: A Modular and Task-Flexible Recommendation Library},
  author={Li, Jiayu and Li, Hanyu and He, Zhiyu and Ma, Weizhi and Sun, Peijie and Zhang, Min and Ma, Shaoping},
  booktitle={Proceedings of the 18th ACM Conference on Recommender Systems},
  pages={454--464},
  year={2024}
}
