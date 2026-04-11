#!/usr/bin/env bash
# M2 Scaling Attack 全量实验启动器
# 用法：
#   bash scripts/run_m2_scaling_full.sh [--dry-run]
#
# 实验矩阵：
#   正式实验 24 组 (gamma=10):
#     CIFAR-10: 4 alphas × 3 seeds × 100 rounds (ResNet18)
#     MNIST:    4 alphas × 3 seeds × 50  rounds (LeNet5)
#   控制实验 3 组 (gamma=1):
#     CIFAR-10: alpha=0.5, seed=0/1/2, 100 rounds (ResNet18)
#
# 在 3 张 GPU (1,2,3) 上并行，每张 GPU 串行跑分配到的实验。
# 每个实验通过 tmux 窗口跑，便于后台运行和日志查看。

set -euo pipefail
# SCRIPT_DIR points to the shieldfl project root (parent of scripts/)
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"
DRY_RUN="false"
if [[ "${1:-}" == "--dry-run" ]]; then
    DRY_RUN="true"
fi

# GPU 分配
GPUS=(1 2 3)

# 全局固定参数
CLIENTS=10
PMR=0.3
EPOCHS=1
BATCH=64
GPU_MAPPING="mapping_single_gpu"
RUNTIME="single-gpu-deterministic"

# 正式实验参数网格
ALPHAS=(0.1 0.3 0.5 100)
SEEDS=(0 1 2)

# 构建实验列表
# 格式: "dataset model rounds alpha seed gamma gpu_id"
declare -a EXPERIMENTS=()

# 正式实验 (gamma=10)
for alpha in "${ALPHAS[@]}"; do
    for seed in "${SEEDS[@]}"; do
        # CIFAR-10
        EXPERIMENTS+=("cifar10 ResNet18 100 $alpha $seed 10")
        # MNIST
        EXPERIMENTS+=("mnist LeNet5 50 $alpha $seed 10")
    done
done

# 控制实验 (gamma=1, CIFAR-10, alpha=0.5)
for seed in "${SEEDS[@]}"; do
    EXPERIMENTS+=("cifar10 ResNet18 100 0.5 $seed 1")
done

TOTAL=${#EXPERIMENTS[@]}
echo "=== M2 Scaling Full Experiment Matrix ==="
echo "Total experiments: $TOTAL"
echo "GPUs: ${GPUS[*]}"
echo ""

# 将实验分配到 GPU (round-robin)
NUM_GPUS=${#GPUS[@]}
declare -a GPU_TASKS_0=()
declare -a GPU_TASKS_1=()
declare -a GPU_TASKS_2=()

for i in "${!EXPERIMENTS[@]}"; do
    gpu_slot=$((i % NUM_GPUS))
    case $gpu_slot in
        0) GPU_TASKS_0+=("${EXPERIMENTS[$i]}") ;;
        1) GPU_TASKS_1+=("${EXPERIMENTS[$i]}") ;;
        2) GPU_TASKS_2+=("${EXPERIMENTS[$i]}") ;;
    esac
done

echo "GPU ${GPUS[0]}: ${#GPU_TASKS_0[@]} experiments"
echo "GPU ${GPUS[1]}: ${#GPU_TASKS_1[@]} experiments"
echo "GPU ${GPUS[2]}: ${#GPU_TASKS_2[@]} experiments"
echo ""

# 生成单个 GPU 的串行运行脚本
generate_gpu_script() {
    local gpu_id=$1
    shift
    local tasks=("$@")
    local script_file="/tmp/m2_scaling_gpu${gpu_id}.sh"

    cat > "$script_file" << 'HEADER'
#!/usr/bin/env bash
set -euo pipefail
HEADER

    cat >> "$script_file" << VARS
GPU_ID=${gpu_id}
SCRIPT_DIR="${SCRIPT_DIR}"
TOTAL_TASKS=${#tasks[@]}
VARS

    local task_idx=0
    for task in "${tasks[@]}"; do
        read -r dataset model rounds alpha seed gamma <<< "$task"
        task_idx=$((task_idx + 1))

        # Determine test_subset and max_samples based on dataset
        local max_samples=0  # 0 means no limit for full experiments
        local test_subset=0  # 0 means no limit

        # For production: remove max_samples & test_subset limits
        # run_experiment.sh defaults: max_samples=300, test_subset=500
        # We need to pass large values to effectively disable the limit

        local gamma_tag=""
        if [[ "$gamma" != "10" ]]; then
            gamma_tag="_g${gamma}"
        fi
        local log_file="/tmp/m2_scaling_${dataset}_${model}_a${alpha}_s${seed}${gamma_tag}.log"

        cat >> "$script_file" << TASK

echo ""
echo "=========================================="
echo "[GPU ${gpu_id}] Task ${task_idx}/${#tasks[@]}: ${dataset} ${model} alpha=${alpha} seed=${seed} gamma=${gamma} rounds=${rounds}"
echo "=========================================="
echo "Start: \$(date '+%Y-%m-%d %H:%M:%S')"

cd "\${SCRIPT_DIR}"
bash "\${SCRIPT_DIR}/scripts/run_experiment.sh" \\
  --model ${model} --dataset ${dataset} \\
  --attack model_replacement --defense none --aggregator fedavg \\
  --pmr ${PMR} --alpha ${alpha} --seed ${seed} \\
  --rounds ${rounds} --clients ${CLIENTS} --epochs ${EPOCHS} --batch_size ${BATCH} \\
  --gpu --gpu_id ${gpu_id} --gpu_mapping ${GPU_MAPPING} --runtime ${RUNTIME} \\
  --scale_gamma ${gamma} \\
  --max_samples 0 --test_subset 0 \\
  2>&1 | tee "${log_file}"

echo "End: \$(date '+%Y-%m-%d %H:%M:%S')"
echo "Exit code: \$?"
TASK
    done

    cat >> "$script_file" << 'FOOTER'

echo ""
echo "=========================================="
echo "All tasks on this GPU completed!"
echo "=========================================="
FOOTER

    chmod +x "$script_file"
    echo "$script_file"
}

# 辅助函数：获取 GPU 任务数组
get_gpu_tasks() {
    local idx=$1
    case $idx in
        0) echo "${GPU_TASKS_0[@]}" ;;
        1) echo "${GPU_TASKS_1[@]}" ;;
        2) echo "${GPU_TASKS_2[@]}" ;;
    esac
}

# 打印实验分配
echo "--- Experiment Assignment ---"
for gpu_idx in 0 1 2; do
    gpu_id=${GPUS[$gpu_idx]}
    echo ""
    echo "GPU $gpu_id:"
    case $gpu_idx in
        0) tasks=("${GPU_TASKS_0[@]}") ;;
        1) tasks=("${GPU_TASKS_1[@]}") ;;
        2) tasks=("${GPU_TASKS_2[@]}") ;;
    esac
    local_idx=0
    for task in "${tasks[@]}"; do
        local_idx=$((local_idx + 1))
        read -r dataset model rounds alpha seed gamma <<< "$task"
        echo "  [$local_idx] ${dataset} ${model} alpha=${alpha} seed=${seed} gamma=${gamma} rounds=${rounds}"
    done
done
echo ""

if [[ "$DRY_RUN" == "true" ]]; then
    echo "[DRY RUN] Generating scripts but not launching..."
    for gpu_idx in 0 1 2; do
        gpu_id=${GPUS[$gpu_idx]}
        case $gpu_idx in
            0) script=$(generate_gpu_script "$gpu_id" "${GPU_TASKS_0[@]}") ;;
            1) script=$(generate_gpu_script "$gpu_id" "${GPU_TASKS_1[@]}") ;;
            2) script=$(generate_gpu_script "$gpu_id" "${GPU_TASKS_2[@]}") ;;
        esac
        echo "  GPU $gpu_id script: $script"
    done
    echo ""
    echo "[DRY RUN] To launch manually:"
    echo "  tmux new-session -d -s m2_gpu1 'bash /tmp/m2_scaling_gpu1.sh'"
    echo "  tmux new-session -d -s m2_gpu2 'bash /tmp/m2_scaling_gpu2.sh'"
    echo "  tmux new-session -d -s m2_gpu3 'bash /tmp/m2_scaling_gpu3.sh'"
    exit 0
fi

# 生成并在 tmux 中启动
TMUX_SESSION_PREFIX="m2_scaling"

for gpu_idx in 0 1 2; do
    gpu_id=${GPUS[$gpu_idx]}
    case $gpu_idx in
        0) script=$(generate_gpu_script "$gpu_id" "${GPU_TASKS_0[@]}") ;;
        1) script=$(generate_gpu_script "$gpu_id" "${GPU_TASKS_1[@]}") ;;
        2) script=$(generate_gpu_script "$gpu_id" "${GPU_TASKS_2[@]}") ;;
    esac

    session_name="${TMUX_SESSION_PREFIX}_gpu${gpu_id}"

    # Kill existing session if any
    tmux kill-session -t "$session_name" 2>/dev/null || true

    echo "Launching GPU $gpu_id in tmux session: $session_name"
    echo "  Script: $script"

    # Activate venv and run
    tmux new-session -d -s "$session_name" \
        "source /data/home/ykdz/FedML/.venv/bin/activate && bash $script 2>&1 | tee /tmp/m2_scaling_gpu${gpu_id}_master.log; echo 'SESSION DONE'; exec bash"
done

echo ""
echo "=== All experiments launched ==="
echo ""
echo "Monitor with:"
echo "  tmux attach -t ${TMUX_SESSION_PREFIX}_gpu1"
echo "  tmux attach -t ${TMUX_SESSION_PREFIX}_gpu2"
echo "  tmux attach -t ${TMUX_SESSION_PREFIX}_gpu3"
echo ""
echo "Check logs:"
echo "  tail -f /tmp/m2_scaling_gpu1_master.log"
echo "  tail -f /tmp/m2_scaling_gpu2_master.log"
echo "  tail -f /tmp/m2_scaling_gpu3_master.log"
echo ""
echo "List sessions: tmux ls | grep ${TMUX_SESSION_PREFIX}"
