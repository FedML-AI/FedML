#!/usr/bin/env bash
# ================================================================
# ShieldFL GPU 实验一键启动（3-GPU 并行 tmux）
#
# 在远端服务器上运行此脚本，将同时启动 3 个 tmux session：
#   m1 → GPU 0 : M1 基线实验 (24 个)
#   m2 → GPU 1 : M2 攻击优先子集 (36 个)
#   m3 → GPU 3 : M3 防御优先子集 (36 个)
#
# 用法：
#   bash scripts/launch_all_gpu.sh          # 启动全部
#   bash scripts/launch_all_gpu.sh m1       # 仅启动 M1
#   bash scripts/launch_all_gpu.sh m2 m3    # 启动 M2 和 M3
#
# 预估时间（RTX 4090）：
#   M1: ~6h (12×CIFAR10@100r + 12×MNIST@50r)
#   M2: ~4h (18×CIFAR10@50r + 18×MNIST@50r)
#   M3: ~6h (18×CIFAR10@50r + 18×MNIST@50r, VeriFL 聚合更慢)
#   三路并行 → 最慢的约 6h
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
VENV="/data/home/ykdz/FedML/.venv/bin/activate"
LOG_DIR="/data/home/ykdz"

# 默认启动全部
TARGETS="${@:-m1 m2 m3}"

# 清理已完成的旧 tmux session（可选）
for t in $TARGETS; do
	if tmux has-session -t "$t" 2>/dev/null; then
		echo "WARNING: tmux session '$t' already exists. Kill it first:"
		echo "  tmux kill-session -t $t"
		exit 1
	fi
done

# 通用 prefix：激活 venv + 进入项目目录
CMD_PREFIX="source ${VENV} && cd ${PROJECT_DIR}"

# 先清理旧的残留 MPI 进程
echo "Cleaning up stale processes..."
pkill -9 -f main_fedml_shieldfl 2>/dev/null || true
sleep 1

for t in $TARGETS; do
	case $t in
	m1)
		echo "Launching M1 (GPU 0) → tmux session 'm1', log: ${LOG_DIR}/m1_gpu.log"
		tmux new-session -d -s m1 \
			"${CMD_PREFIX} && GPU_ID=0 bash scripts/run_m1_baseline_gpu.sh 2>&1 | tee ${LOG_DIR}/m1_gpu.log; echo '>>> M1 FINISHED <<<'"
		;;
	m2)
		echo "Launching M2 (GPU 1) → tmux session 'm2', log: ${LOG_DIR}/m2_gpu.log"
		tmux new-session -d -s m2 \
			"${CMD_PREFIX} && GPU_ID=1 bash scripts/run_m2_priority_gpu.sh 2>&1 | tee ${LOG_DIR}/m2_gpu.log; echo '>>> M2 FINISHED <<<'"
		;;
	m3)
		echo "Launching M3 (GPU 3) → tmux session 'm3', log: ${LOG_DIR}/m3_gpu.log"
		tmux new-session -d -s m3 \
			"${CMD_PREFIX} && GPU_ID=3 bash scripts/run_m3_priority_gpu.sh 2>&1 | tee ${LOG_DIR}/m3_gpu.log; echo '>>> M3 FINISHED <<<'"
		;;
	*)
		echo "Unknown target: $t (use m1, m2, m3)"
		exit 1
		;;
	esac
done

echo ""
echo "=== All requested sessions launched ==="
tmux ls
echo ""
echo "快速命令："
echo "  查看 M1: tmux attach -t m1"
echo "  查看 M2: tmux attach -t m2"
echo "  查看 M3: tmux attach -t m3"
echo "  脱离 tmux: Ctrl+B, D"
echo "  监控进度: bash scripts/check_progress.sh"
