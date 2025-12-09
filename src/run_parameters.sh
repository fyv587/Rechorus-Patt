#!/usr/bin/env bash
set -e

# ================= 配置区域 =================
MODEL="PAtt"
GPU_ID=0
# 您想要测试的数据集
DATASETS=("MovieLens_1M" "Grocery_and_Gourmet_Food" "MIND_Large")

# === 默认基准参数 (Base Settings) ===
# 当测试某一个参数时，其他参数保持在这个默认值
BASE_LR=1e-4
BASE_EMB=64
BASE_DROP=0.3
BASE_LAYER=1
BASE_HEAD=4

# === 探索范围 (根据论文 A.2 章节) ===
# 1. Dropout (丢弃率)
DROP_LIST=(0.3 0.5 0.7)

# 2. Embedding Size (嵌入维度)
EMB_LIST=(32 64 128)

# 3. Learning Rate (学习率)
LR_LIST=(1e-3 1e-4)

# 4. Structure - Layers (层数)
LAYER_LIST=(1 2 3)

# 5. Structure - Heads (头数)
HEAD_LIST=(1 2 4)

# 路径设置
MAIN_PY="main.py"
DATA_PATH="../data/"
LOG_ROOT="../logs_hyper_general"

mkdir -p "${LOG_ROOT}"

# ================= 训练函数 =================
run_exp() {
    local dataset=$1
    local exp_name=$2  # 实验名称，如 drop_0.5
    local lr=$3
    local emb=$4
    local drop=$5
    local layer=$6
    local head=$7

    # 检查合法性: embedding size 必须能被 num_heads 整除
    if (( emb % head != 0 )); then
        echo ">>> [SKIP] Invalid config: emb=${emb} is not divisible by head=${head}"
        return
    fi

    # 定义日志文件路径
    local log_dir="${LOG_ROOT}/${MODEL}/${dataset}"
    mkdir -p "${log_dir}"
    local log_file="${log_dir}/${exp_name}.log"

    # 如果日志已存在，跳过（避免重复跑）
    if [ -f "$log_file" ]; then
        echo ">>> [SKIP] Log exists: ${log_file}"
        return
    fi

    echo "--------------------------------------------------------------------------------"
    echo ">>> Running [${dataset}]: ${exp_name}"
    echo "    Params: LR=${lr}, Emb=${emb}, Drop=${drop}, Layer=${layer}, Head=${head}"
    echo "    Log -> ${log_file}"

    # 运行命令 (已移除 --patt_lambda)
    CMD="python ${MAIN_PY} \
      --model_name ${MODEL} \
      --dataset ${dataset} \
      --history_max 30 \
      --batch_size 256 \
      --eval_batch_size 256 \
      --early_stop 10 \
      --metric NDCG,HR \
      --topk 5,20 \
      --gpu ${GPU_ID} \
      --path ${DATA_PATH} \
      --lr ${lr} \
      --emb_size ${emb} \
      --dropout ${drop} \
      --num_layers ${layer} \
      --num_heads ${head}"

    # 执行并保存日志
    eval "${CMD}" > "${log_file}" 2>&1
}

# ================= 开始循环实验 =================

for dataset in "${DATASETS[@]}"; do
    echo "################################################################"
    echo "Starting experiments for Dataset: ${dataset}"
    echo "################################################################"

    # --- Group 1: 测试 Dropout ---
    # 固定其他参数为 Base，只变 Dropout
    for drop in "${DROP_LIST[@]}"; do
        run_exp "$dataset" "dropout_${drop}" \
            "$BASE_LR" "$BASE_EMB" "$drop" "$BASE_LAYER" "$BASE_HEAD"
    done

    # --- Group 2: 测试 Embedding Size ---
    # 固定其他参数为 Base，只变 Emb Size
    for emb in "${EMB_LIST[@]}"; do
        run_exp "$dataset" "emb_${emb}" \
            "$BASE_LR" "$emb" "$BASE_DROP" "$BASE_LAYER" "$BASE_HEAD"
    done

    # --- Group 3: 测试 Learning Rate ---
    # 固定其他参数为 Base，只变 LR
    for lr in "${LR_LIST[@]}"; do
        run_exp "$dataset" "lr_${lr}" \
            "$lr" "$BASE_EMB" "$BASE_DROP" "$BASE_LAYER" "$BASE_HEAD"
    done

    # --- Group 4: 测试 Layers (Transformer深度) ---
    # 固定其他参数为 Base，只变 Layer
    for layer in "${LAYER_LIST[@]}"; do
        run_exp "$dataset" "layer_${layer}" \
            "$BASE_LR" "$BASE_EMB" "$BASE_DROP" "$layer" "$BASE_HEAD"
    done

    # --- Group 5: 测试 Heads (注意力头数) ---
    # 固定其他参数为 Base，只变 Head
    for head in "${HEAD_LIST[@]}"; do
        run_exp "$dataset" "head_${head}" \
            "$BASE_LR" "$BASE_EMB" "$BASE_DROP" "$BASE_LAYER" "$head"
    done
    
    echo ""
done

echo "All experiments finished. Results are in ${LOG_ROOT}"