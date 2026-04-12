# N=50 FedAvg Baseline Archive

本目录是对 PR `#2263`（`Exp/n50 fedavg baseline 20260411`）的整理入库版本，只保留 **N=50 FedAvg 无攻击基线** 的可复现实验资产。

## 内容概览

- `27_[2026-04-11]_N50_FedAvg_无攻击基线实验设计.md`：实验设计与验收标准归档版。
- `28_[2026-04-12]_N50_FedAvg_实验结果与验收归档说明.md`：结果汇总、验收结论与对照分析。
- `n50_results/`：30 组实验的原始 metrics / configs 快照。
- `tools/analyze_baseline.py`：用于复盘统计与 AC 检查的离线分析脚本。
- `../python/examples/federate/prebuilt_jobs/shieldfl/scripts/batch_baseline_n50.sh`：批量执行脚本。

## 整理原则

- 排除 PR 中与本次基线无关的临时草稿、过程记录和工作台文件。
- 保留完整的 `2 数据集 × 3 α × 5 seeds = 30` 组实验矩阵。
- 原始文件名保持不变，以便与报告、脚本和历史运行结果逐项对照。

## 命名说明

- 原始 JSONL / YAML 文件名沿用现有实验管线的 `shieldfl` 命名；在这批实验里，它表示由 ShieldFL 实验框架产出的 **no-attack / no-defense 基线记录**，并不表示额外启用了某种防御。
- 指标文件中的 `gauto` 后缀也保留原样；由于本实验 `attack=none`，文件内 `gamma_actual` 为 `null`，该后缀仅用于保持与 PR 原始资产一致。
- `config_ResNet18_cifar10_shieldfl_atknone_defnone_a0.5_pmr0.0_seed0.yaml` 在 `master` 上已存在，因此未出现在 PR diff 中；本次归档时已将其补齐，以完整覆盖 30 组矩阵。

## 关联路径

- 原始执行脚本：`python/examples/federate/prebuilt_jobs/shieldfl/scripts/batch_baseline_n50.sh`
- 原始运行结果根目录：`python/examples/federate/prebuilt_jobs/shieldfl/results/`
- 原始 PR 分支：`YKDZ/FedML:exp/n50-fedavg-baseline-20260411`
