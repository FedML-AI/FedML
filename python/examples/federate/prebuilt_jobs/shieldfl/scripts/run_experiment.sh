#!/usr/bin/env bash
# 使用方法：
#   bash scripts/run_experiment.sh \
#     --model ResNet18 --dataset cifar10 \
#     --attack none --defense none \
#     --pmr 0.0 --alpha 0.5 --seed 0 \
#     --rounds 3 --clients 5 --epochs 1 --batch_size 32

set -euo pipefail
SCRIPT_DIR="$(cd "$(dirname "$0")/.." && pwd)"

# ----------- 参数默认值 -----------
MODEL="SimpleCNN"
DATASET="cifar10"
ATTACK="none"
DEFENSE="none"
PMR="0.0"
ALPHA="0.5"
SEED="0"
ROUNDS="3"
CLIENTS="3"
EPOCHS="1"
BATCH="32"
MAX_SAMPLES="300"
VAL_PER_CLASS="50"
TEST_SUBSET="500"
GPU="false"
GPU_ID="0"
RUNTIME_MODE="cpu-deterministic"
GPU_MAPPING_KEY="mapping_default"
CPU_TRANSFER="true"
WEIGHT_DECAY="auto"
SERVER_LR="1.0"
LR="0.01"
SCALE_GAMMA="auto"
BACKDOOR_PER_BATCH="20"
ATTACK_ROUNDS=""
GPU_PROC_MAPPING=""

while [[ $# -gt 0 ]]; do
	case $1 in
	--model)
		MODEL="$2"
		shift 2
		;;
	--dataset)
		DATASET="$2"
		shift 2
		;;
	--attack)
		ATTACK="$2"
		shift 2
		;;
	--defense)
		DEFENSE="$2"
		shift 2
		;;
	--pmr)
		PMR="$2"
		shift 2
		;;
	--alpha)
		ALPHA="$2"
		shift 2
		;;
	--seed)
		SEED="$2"
		shift 2
		;;
	--rounds)
		ROUNDS="$2"
		shift 2
		;;
	--clients)
		CLIENTS="$2"
		shift 2
		;;
	--epochs)
		EPOCHS="$2"
		shift 2
		;;
	--batch_size)
		BATCH="$2"
		shift 2
		;;
	--max_samples)
		MAX_SAMPLES="$2"
		shift 2
		;;
	--test_subset)
		TEST_SUBSET="$2"
		shift 2
		;;
	--gpu)
		GPU="true"
		shift 1
		;;
	--gpu_id)
		GPU_ID="$2"
		shift 2
		;;
	--runtime)
		RUNTIME_MODE="$2"
		shift 2
		;;
	--gpu_mapping)
		GPU_MAPPING_KEY="$2"
		shift 2
		;;
	--cpu_transfer)
		CPU_TRANSFER="$2"
		shift 2
		;;
	--weight_decay)
		WEIGHT_DECAY="$2"
		shift 2
		;;
	--server_lr)
		SERVER_LR="$2"
		shift 2
		;;
	--lr)
		LR="$2"
		shift 2
		;;
	--scale_gamma)
		SCALE_GAMMA="$2"
		shift 2
		;;
	--backdoor_per_batch)
		BACKDOOR_PER_BATCH="$2"
		shift 2
		;;
	--attack_rounds)
		ATTACK_ROUNDS="$2"
		shift 2
		;;
	--gpu_proc_mapping)
		GPU_PROC_MAPPING="$2"
		shift 2
		;;
	*)
		echo "Unknown argument: $1"
		exit 1
		;;
	esac
done

# GPU 训练时 MPI 通信仍走 CPU tensor
if [[ "$GPU" == "true" ]]; then
	CPU_TRANSFER="true"
	# runtime_mode 未被用户显式覆写时，自动升级为 GPU 确定性模式
	if [[ "$RUNTIME_MODE" == "cpu-deterministic" ]]; then
		RUNTIME_MODE="single-gpu-deterministic"
	fi
fi

# 自动推导 weight_decay：CIFAR-10 用 1e-4，MNIST 用 0
if [[ "$WEIGHT_DECAY" == "auto" ]]; then
	if [[ "$DATASET" == "cifar10" ]]; then
		WEIGHT_DECAY="0.0001"
	else
		WEIGHT_DECAY="0.0"
	fi
fi

# ----------- 计算攻击参数 -----------
ENABLE_ATTACK="false"
ATTACK_TYPE="none"
BYZANTINE_NUM=0
EVAL_ASR="false"

if [[ "$ATTACK" != "none" ]]; then
	ENABLE_ATTACK="true"
	ATTACK_TYPE="$ATTACK"
	BYZANTINE_NUM=$(python3 -c "import math; print(max(1, math.ceil($CLIENTS * $PMR)))")
	if [[ "$ATTACK" == "model_replacement" ]]; then
		EVAL_ASR="true"
		# D-2: Default to every-round attack (null) for model_replacement.
		# Override with --attack_rounds '[95,96,97,98,99]' for specific rounds.
		if [[ -z "$ATTACK_ROUNDS" ]]; then
			ATTACK_ROUNDS="null"
		fi
	fi
fi

ENABLE_DEFENSE="false"
DEFENSE_TYPE="none"
TRIM_BETA="0.2"
if [[ "$DEFENSE" != "none" ]]; then
	DEFENSE_TYPE="$DEFENSE"
	# "shieldfl" defense is implemented in the custom aggregator, not FedML's FedMLDefender.
	# Only enable FedML's built-in defender for recognized defense types.
	if [[ "$DEFENSE" != "shieldfl" ]]; then
		ENABLE_DEFENSE="true"
	fi
fi

# ----------- 生成临时配置 -----------

# Build attack-specific YAML fields
ATTACK_EXTRA_YAML=""
if [[ "$ATTACK" == "model_replacement" ]]; then
	ATTACK_EXTRA_YAML="  scale_gamma: ${SCALE_GAMMA}
  attack_training_rounds: ${ATTACK_ROUNDS}
  backdoor_per_batch: ${BACKDOOR_PER_BATCH}
  attacker_epochs: null
  attacker_lr: null
  attacker_weight_decay: null
  attacker_noise_sigma: 0"
elif [[ "$ATTACK" != "none" ]]; then
	ATTACK_EXTRA_YAML="  attack_mode: \"flip\""
fi

CONFIG_FILE="/tmp/shieldfl_exp_${MODEL}_${DATASET}_${ATTACK}_${DEFENSE}_a${ALPHA}_pmr${PMR}_s${SEED}.yaml"
WORKER_NUM=$CLIENTS

cat >"$CONFIG_FILE" <<EOF
common_args:
  training_type: "cross_silo"
  random_seed: ${SEED}

data_args:
  dataset: "${DATASET}"
  data_cache_dir: ./data
  partition_method: "hetero"
  partition_alpha: ${ALPHA}
  val_per_class: ${VAL_PER_CLASS}
  trust_per_class: ${VAL_PER_CLASS}
  max_samples_per_client: ${MAX_SAMPLES}
  test_subset_size: ${TEST_SUBSET}
  num_workers: 0

model_args:
  model: "${MODEL}"

train_args:
  federated_optimizer: "FedAvg"
  client_id_list:
  client_num_in_total: ${CLIENTS}
  client_num_per_round: ${CLIENTS}
  comm_round: ${ROUNDS}
  epochs: ${EPOCHS}
  batch_size: ${BATCH}
  client_optimizer: sgd
  learning_rate: ${LR}
  weight_decay: ${WEIGHT_DECAY}
  momentum: 0.9
  server_momentum: 0.0
  server_lr: ${SERVER_LR}
  pop_size: 15
  generations: 10
  lambda_reg: 0.01
  cpu_transfer: ${CPU_TRANSFER}
  enable_attack: ${ENABLE_ATTACK}
  attack_type: "${ATTACK_TYPE}"
  byzantine_client_num: ${BYZANTINE_NUM}
  enable_defense: ${ENABLE_DEFENSE}
  defense_type: "${DEFENSE_TYPE}"
  beta: ${TRIM_BETA}
  eval_asr: ${EVAL_ASR}
  target_label: 0
  trigger_size: 3
  trigger_value: 1.0
  original_class_list: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  target_class_list: [9, 8, 7, 6, 5, 4, 3, 2, 1, 0]
  ratio_of_poisoned_client: ${PMR}
${ATTACK_EXTRA_YAML}

validation_args:
  frequency_of_the_test: 1

device_args:
  worker_num: ${WORKER_NUM}
  using_gpu: ${GPU}
  gpu_mapping_file: config/gpu_mapping.yaml
  gpu_mapping_key: ${GPU_MAPPING_KEY}

comm_args:
  backend: "MPI"
  is_mobile: 0

tracking_args:
  log_file_dir: ./log
  enable_wandb: false
  using_mlops: false

shieldfl_args:
  runtime_mode: "${RUNTIME_MODE}"
  enforce_determinism: true
  sort_client_updates: true
  aggregator_type: "shieldfl"
  metrics_output_dir: "./results"
EOF

echo "=== ShieldFL Experiment ==="
echo "  model=${MODEL} dataset=${DATASET} attack=${ATTACK} defense=${DEFENSE}"
echo "  pmr=${PMR} alpha=${ALPHA} seed=${SEED}"
echo "  rounds=${ROUNDS} clients=${CLIENTS} epochs=${EPOCHS} lr=${LR}"
echo "  weight_decay=${WEIGHT_DECAY} server_lr=${SERVER_LR}"
echo "  gpu=${GPU} runtime=${RUNTIME_MODE} gpu_mapping=${GPU_MAPPING_KEY}"
echo "  config=${CONFIG_FILE}"

# WI-9: 持久化 YAML 配置副本
PERSIST_DIR="./results/configs"
mkdir -p "$PERSIST_DIR"
PERSIST_NAME="config_${MODEL}_${DATASET}_shieldfl_atk${ATTACK}_def${DEFENSE}_a${ALPHA}_pmr${PMR}_seed${SEED}.yaml"
cp "$CONFIG_FILE" "${PERSIST_DIR}/${PERSIST_NAME}"
echo "  persisted_config=${PERSIST_DIR}/${PERSIST_NAME}"

cd "$SCRIPT_DIR"
TOTAL_PROC=$((WORKER_NUM + 1))

MPI_EXTRA_ARGS="--oversubscribe"
# 允许 root 用户运行 mpirun
if [[ "$(id -u)" == "0" ]]; then
	MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS} --allow-run-as-root"
fi

MPI_CMD="python main_fedml_shieldfl.py"
# GPU 模式下的显存隔离和设备分配
if [[ "$GPU" == "true" ]]; then
	export PYTORCH_CUDA_ALLOC_CONF="expandable_segments:True"
	MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS} -x PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True"
	if [[ -n "$GPU_PROC_MAPPING" ]]; then
		# Per-process GPU isolation: each MPI rank sees only its assigned GPU
		MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS} -x GPU_PROC_MAPPING=${GPU_PROC_MAPPING} -x GPU_PHYS_IDS=${GPU_ID}"
		MPI_CMD="bash scripts/gpu_wrapper.sh main_fedml_shieldfl.py"
	else
		export CUDA_VISIBLE_DEVICES="${GPU_ID}"
		MPI_EXTRA_ARGS="${MPI_EXTRA_ARGS} -x CUDA_VISIBLE_DEVICES=${GPU_ID}"
	fi
fi

MPI_EXIT=0
mpirun ${MPI_EXTRA_ARGS} -np $TOTAL_PROC ${MPI_CMD} --cf "$CONFIG_FILE" || MPI_EXIT=$?
if [[ $MPI_EXIT -ne 0 ]]; then
	echo "WARNING: mpirun exited with code $MPI_EXIT"
fi
exit $MPI_EXIT
