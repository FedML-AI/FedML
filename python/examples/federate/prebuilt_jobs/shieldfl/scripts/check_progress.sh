#!/usr/bin/env bash
# ================================================================
# ShieldFL GPU 实验进度监控
#
# 用法：
#   bash scripts/check_progress.sh           # 完整报告
#   bash scripts/check_progress.sh --brief   # 精简报告
# ================================================================
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
PROJECT_DIR="$(cd "$SCRIPT_DIR/.." && pwd)"
RESULTS_DIR="${PROJECT_DIR}/results"
LOG_DIR="/data/home/ykdz"

BRIEF=false
[[ "${1:-}" == "--brief" ]] && BRIEF=true

echo "=============================================="
echo " ShieldFL GPU 实验进度报告"
echo " $(date '+%Y-%m-%d %H:%M:%S')"
echo "=============================================="

# --- tmux session 状态 ---
echo ""
echo "【tmux Session 状态】"
for s in m1 m2 m3; do
	if tmux has-session -t "$s" 2>/dev/null; then
		echo "  $s: ✅ Running"
	else
		echo "  $s: ⬜ Not running"
	fi
done

# --- GPU 状态 ---
echo ""
echo "【GPU 状态】"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu --format=csv,noheader 2>/dev/null || echo "  nvidia-smi failed"

# --- 结果文件统计 ---
echo ""
echo "【结果文件统计】"
TOTAL_FILES=$(find "$RESULTS_DIR" -name "metrics_*.jsonl" 2>/dev/null | wc -l)
echo "  总文件数: $TOTAL_FILES"

M1_FILES=$(find "$RESULTS_DIR" -name "metrics_*_fedavg_atknone_defnone_*.jsonl" 2>/dev/null | wc -l)
M2_FILES=$(find "$RESULTS_DIR" -name "metrics_*_fedavg_atk*_defnone_*.jsonl" ! -name "*atknone*" 2>/dev/null | wc -l)
M3_DEF=$(find "$RESULTS_DIR" -name "metrics_*_fedavg_atk*_def*.jsonl" ! -name "*defnone*" 2>/dev/null | wc -l)
M3_VERIFL=$(find "$RESULTS_DIR" -name "metrics_*_verifl_atk*.jsonl" 2>/dev/null | wc -l)

echo "  M1 (baseline):       $M1_FILES / 24"
echo "  M2 (attacks):        $M2_FILES / 36  (priority subset)"
echo "  M3-defense (FedML):  $M3_DEF / 24  (priority subset)"
echo "  M3-defense (VeriFL): $M3_VERIFL / 12  (priority subset)"

# --- 日志中的完成标记 ---
echo ""
echo "【日志完成标记】"
for label in m1 m2 m3; do
	LOG="${LOG_DIR}/${label}_gpu.log"
	if [[ -f "$LOG" ]]; then
		if grep -q "DONE at\|FINISHED" "$LOG" 2>/dev/null; then
			DONE_TIME=$(grep -oP "DONE at \K.*|FINISHED" "$LOG" | tail -1)
			echo "  $label: ✅ COMPLETED ($DONE_TIME)"
		else
			# 显示最近的实验标签
			LAST=$(grep -E '\[M[123]-' "$LOG" | tail -1)
			ROUNDS=$(grep -c "end.*round training" "$LOG" 2>/dev/null)
			echo "  $label: 🔄 Running | rounds_done=$ROUNDS | $LAST"
		fi
	else
		echo "  $label: ⬜ No log file"
	fi
done

# --- 详细结果（非 brief 模式）---
if ! $BRIEF && [[ $TOTAL_FILES -gt 0 ]]; then
	echo ""
	echo "【已完成实验详情】"
	echo "  File | Rounds | Final_MA | AggTime"
	echo "  -----|--------|----------|--------"
	find "$RESULTS_DIR" -name "metrics_*.jsonl" -print0 2>/dev/null | sort -z | while IFS= read -r -d '' f; do
		BASENAME=$(basename "$f")
		LAST_LINE=$(tail -1 "$f" 2>/dev/null)
		if [[ -n "$LAST_LINE" ]]; then
			python3 -c "
import json, sys
d = json.loads(sys.argv[1])
print('  %-60s | %3d | %.4f | %.4fs' % (
    sys.argv[2], d.get('round',0), d.get('test_accuracy',0), d.get('agg_time',0)))
" "$LAST_LINE" "$BASENAME" 2>/dev/null || echo "  $BASENAME: parse error"
		fi
	done
fi

# --- 全部完成判定 ---
echo ""
ALL_DONE=true
for label in m1 m2 m3; do
	LOG="${LOG_DIR}/${label}_gpu.log"
	if [[ -f "$LOG" ]] && grep -q "DONE at\|FINISHED" "$LOG" 2>/dev/null; then
		: # ok
	else
		ALL_DONE=false
	fi
done

if $ALL_DONE; then
	echo "🎉 所有实验已完成！运行以下命令生成汇总："
	echo "  cd $PROJECT_DIR && python3 scripts/summarize_results.py"
else
	echo "⏳ 实验仍在运行中。再次检查："
	echo "  bash scripts/check_progress.sh --brief"
fi
echo ""
