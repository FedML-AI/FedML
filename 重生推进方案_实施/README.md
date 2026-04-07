# 重生推进方案 — 实施文档集

本目录汇集了 VeriFL/ShieldFL M0→M1a→M1b 重构实施所需的全部参考文档，作为工程交付的完整文档包。

## 文档索引

### 核心规划

| 文档 | 说明 |
|------|------|
| [重生推进方案.md](重生推进方案.md) | 主实施计划（§0-§9 + 附录），涵盖架构重构、LF 回归、Scaling 验证 |
| [VeriFL_v16_to_v18f_升级方案.md](VeriFL_v16_to_v18f_升级方案.md) | VeriFL 算法升级差异分析：v16→v18f 的完整对比和代码级变更清单 |

### 算法规格

| 文档 | 说明 |
|------|------|
| [algrothm.md](algrothm.md) | VeriFL-v18f 完整算法说明（来自 ShieldFL-main/Flower 框架，17 节，含伪代码） |

### 审计与根因分析

| 文档 | 说明 |
|------|------|
| [ShieldFL_基础设施审计报告.md](ShieldFL_基础设施审计报告.md) | 基础设施审计：28 项发现（5 Fatal, 5 Severe, 10 Moderate, 5 Low, 3 Info） |
| [M2_LF_AUDIT_REPORT.md](M2_LF_AUDIT_REPORT.md) | M2 Label Flipping 审计报告（7 缺陷全部修复） |
| [Scaling_失败根因分析.md](Scaling_失败根因分析.md) | Scaling Attack 实验失败根因（γ=N 公式溢出 + VeriFL≠FedAvg） |

### 攻击实施规格

| 文档 | 说明 |
|------|------|
| [Scaling_实施定稿.md](Scaling_实施定稿.md) | Scaling Attack 实施定稿 |
| [LF_实施规格.md](LF_实施规格.md) | Label Flipping 实施规格 |
| [威胁模型.md](威胁模型.md) | 威胁模型定义 |
| [attack_list.md](attack_list.md) | 攻击类型清单 |

## 关键交叉引用

- 重生推进方案 **W-M0-1** → 引用 VeriFL_v16_to_v18f_升级方案.md
- 重生推进方案 **D-2** (移除 BN recal) → v18f 升级方案 §5 一致
- 重生推进方案 **D-3** (BN-aware momentum) → v18f 升级方案 §4 一致
- v18f 升级方案 **C-2∼C-8** → 覆盖重生推进方案 W-M0-1 的全部代码变更
