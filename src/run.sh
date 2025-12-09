#!/usr/bin/env bash
set -e  # 出错立即退出

# 你要测试的模型列表
MODELS=(
  "Caser"
  "FPMC"
  "KDA"
  "SLRCPlus"
  "TiMiRec"
  "TiSASRec"
)

# 你要测试的数据集列表
DATASETS=(
  "MovieLens_1M"
  "Grocery_and_Gourmet_Food"
  "MIND_Large"
)

# 公共超参数
LR=1e-4
EMB_SIZE=64
HISTORY_MAX=30
DROPOUT=0.3
NUM_LAYERS=1
NUM_HEADS=4
BATCH_SIZE=256
EVAL_BATCH_SIZE=256
EARLY_STOP=10
METRIC="NDCG,HR"
TOPK="5,20"
GPU_ID=0

# main.py 和 run.sh 在同一目录
MAIN_PY="main.py"

# data 路径（相对于 main.py）
DATA_PATH="../data/"

# 日志目录（在项目根目录创建 logs/）
LOG_ROOT="../logs"

mkdir -p "${LOG_ROOT}"

for model in "${MODELS[@]}"; do
  for dataset in "${DATASETS[@]}"; do

    # logs/<model>/<dataset>/
    EXP_DIR="${LOG_ROOT}/${model}/${dataset}"
    mkdir -p "${EXP_DIR}"

    echo "====================================================="
    echo "Running experiment: model=${model}, dataset=${dataset}"
    echo "Logs -> ${EXP_DIR}/train.log"
    echo "====================================================="

    CMD="python ${MAIN_PY} \
      --model_name ${model} \
      --dataset ${dataset} \
      --lr ${LR} \
      --emb_size ${EMB_SIZE} \
      --history_max ${HISTORY_MAX} \
      --dropout ${DROPOUT} \
      --num_layers ${NUM_LAYERS} \
      --num_heads ${NUM_HEADS} \
      --batch_size ${BATCH_SIZE} \
      --eval_batch_size ${EVAL_BATCH_SIZE} \
      --early_stop ${EARLY_STOP} \
      --metric ${METRIC} \
      --topk ${TOPK} \
      --gpu ${GPU_ID} \
      --path ${DATA_PATH}
    "

    echo "${CMD}"

    CUDA_VISIBLE_DEVICES=${GPU_ID} ${CMD} 2>&1 | tee "${EXP_DIR}/train.log"

    echo "Finished: model=${model}, dataset=${dataset}"
    echo
  done
done

echo "All experiments finished."
