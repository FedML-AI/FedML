# N=50 FedAvg Baseline Results Archive

本目录存放从 PR `#2263` 整理入库的 **N=50 FedAvg 无攻击基线** 原始结果快照。

## 来源

- 来源 PR：`#2263 Exp/n50 fedavg baseline 20260411`
- 来源分支：`YKDZ/FedML:exp/n50-fedavg-baseline-20260411`
- 整理日期：`2026-04-12`
- 原始结果根目录：`python/examples/federate/prebuilt_jobs/shieldfl/results/`
- 完整性校验：
  - `30/30` 个 metrics 文件齐全
  - 每个 metrics 文件均为 `100` 行，末轮 `round=99`
  - `30/30` 个 configs 文件齐全

## 目录说明

- `metrics/`：30 个原始 JSONL 指标文件，对应 30 组 baseline run。
- `configs/`：30 个 YAML 配置快照，用于参数审计与可复现性核对。

## 使用注意

- 文件名保留原始 `shieldfl` / `gauto` 命名，以维持与报告、脚本和 PR 证据链的一致性。
- 本归档只保留最终需要复核的原始 metrics / configs；运行期 `batch_logs/`、`done.txt` 等过程文件未纳入仓库。
- `config_ResNet18_cifar10_shieldfl_atknone_defnone_a0.5_pmr0.0_seed0.yaml` 来自主分支现有文件，用于补齐完整的 30 组实验矩阵；其余 29 个 config 和 30 个 metrics 来自 PR 分支。

## 关联文档

- `doc_fedavg/27_[2026-04-11]_N50_FedAvg_无攻击基线实验设计.md`
- `doc_fedavg/28_[2026-04-12]_N50_FedAvg_实验结果与验收归档说明.md`
- `doc_fedavg/tools/analyze_baseline.py`
