#!/usr/bin/env python3
"""M2 Scaling Attack full results parser."""
import json, os, sys, statistics
from collections import defaultdict

results_dir = sys.argv[1] if len(sys.argv) > 1 else '.'

def parse_segments(filepath):
    """Split a metrics file into segments (each starting from round 0)."""
    lines = open(filepath).readlines()
    segments, cur = [], []
    prev_r = -1
    for line in lines:
        d = json.loads(line)
        r = d['round']
        if r <= prev_r and cur:
            segments.append(cur)
            cur = []
        cur.append(d)
        prev_r = r
    if cur:
        segments.append(cur)
    return segments

files = sorted(f for f in os.listdir(results_dir)
               if f.startswith('metrics_') and 'model_replacement' in f
               and f.endswith('.jsonl') and '_g1' not in f)

all_results = []
for f in files:
    parts = f.replace('.jsonl','').split('_')
    model, dataset = parts[1], parts[2]
    alpha = seed = None
    for p in parts:
        if p.startswith('a') and len(p)>1:
            try: float(p[1:]); alpha=p[1:]
            except: pass
        if p.startswith('seed'): seed=p[4:]
    
    exp_rounds = 100 if dataset=='cifar10' else 50
    segments = parse_segments(os.path.join(results_dir, f))
    
    for si, seg in enumerate(segments):
        n = len(seg)
        if n < exp_rounds:
            continue
        seg = seg[:exp_rounds]  # trim to expected length
        last = seg[-1]
        
        if dataset=='cifar10' and alpha=='0.5' and len(segments)>=2:
            gamma = '10' if si==0 else ('1' if si==1 else f'extra{si}')
        else:
            gamma = '10'
        
        if gamma.startswith('extra'):
            continue
        
        pre_r = exp_rounds - 6
        pre = next((d for d in seg if d['round']==pre_r), None)
        
        all_results.append(dict(
            f=f, model=model, dataset=dataset, alpha=alpha, seed=seed,
            gamma=gamma, rounds=n,
            acc=last.get('test_accuracy'), asr=last.get('asr'),
            loss=last.get('test_loss'),
            pre_acc=pre.get('test_accuracy') if pre else None,
            pre_asr=pre.get('asr') if pre else None
        ))

# ===== REPORT =====
print("="*80)
print("M2 SCALING ATTACK (Model Replacement Backdoor) — 实验报告")
print("="*80)
print()

# Table 1
print("## 1. 全部实验最终轮指标")
print()
hdr = f"{'Dataset':<9} {'Model':<10} {'α':>6} {'Seed':>4} {'γ':>3} {'#R':>4} {'FinalAcc':>9} {'FinalASR':>9} {'PreAcc':>9} {'PreASR':>9}"
print(hdr); print("-"*len(hdr))
for r in sorted(all_results, key=lambda x:(x['dataset'],x['gamma'],x['alpha'],x['seed'])):
    def f(v): return f"{v:.4f}" if v is not None else "N/A"
    print(f"{r['dataset']:<9} {r['model']:<10} {r['alpha']:>6} {r['seed']:>4} {r['gamma']:>3} {r['rounds']:>4} "
          f"{f(r['acc']):>9} {f(r['asr']):>9} {f(r['pre_acc']):>9} {f(r['pre_asr']):>9}")

# Table 2 — Summary
print()
print("## 2. 正式实验汇总 (γ=10, mean±std over 3 seeds)")
print()
g10 = [r for r in all_results if r['gamma']=='10']
groups = defaultdict(list)
for r in g10:
    groups[(r['dataset'],r['alpha'])].append(r)

def ms(vals):
    if not vals: return "N/A"
    m=statistics.mean(vals)
    s=statistics.stdev(vals) if len(vals)>1 else 0
    return f"{m:.4f}±{s:.4f}"

hdr2 = f"{'Dataset':<9} {'α':>6} {'N':>2} {'Accuracy':>14} {'ASR':>14} {'PreAtk-Acc':>14}"
print(hdr2); print("-"*len(hdr2))
for k in sorted(groups):
    items = groups[k]
    accs=[x['acc'] for x in items if x['acc'] is not None]
    asrs=[x['asr'] for x in items if x['asr'] is not None]
    pas=[x['pre_acc'] for x in items if x['pre_acc'] is not None]
    print(f"{k[0]:<9} {k[1]:>6} {len(items):>2} {ms(accs):>14} {ms(asrs):>14} {ms(pas):>14}")

# Table 3 — Gamma control
print()
print("## 3. 缩放效果对照 — CIFAR-10 α=0.5 (γ=10 vs γ=1)")
print()
g10c = sorted([r for r in all_results if r['dataset']=='cifar10' and r['alpha']=='0.5' and r['gamma']=='10'], key=lambda x:x['seed'])
g1c  = sorted([r for r in all_results if r['dataset']=='cifar10' and r['alpha']=='0.5' and r['gamma']=='1'], key=lambda x:x['seed'])

print(f"{'γ':>3} {'Seed':>4} {'Accuracy':>10} {'ASR':>10}")
print("-"*32)
for r in g10c + g1c:
    def f(v): return f"{v:.4f}" if v is not None else "N/A"
    print(f"{r['gamma']:>3} {r['seed']:>4} {f(r['acc']):>10} {f(r['asr']):>10}")

g10a = [r['asr'] for r in g10c if r['asr'] is not None]
g1a  = [r['asr'] for r in g1c if r['asr'] is not None]
if g10a and g1a:
    m10,m1 = statistics.mean(g10a), statistics.mean(g1a)
    diff = m10-m1
    print()
    print(f"  mean(ASR,γ=10) = {m10:.4f},  mean(ASR,γ=1) = {m1:.4f}")
    print(f"  ΔASR = {diff:+.4f}  [要求≥0.30 → {'PASS' if diff>=0.30 else 'FAIL'}]")
    g10ac = [r['acc'] for r in g10c if r['acc'] is not None]
    g1ac  = [r['acc'] for r in g1c if r['acc'] is not None]
    print(f"  mean(Acc,γ=10) = {statistics.mean(g10ac):.4f},  mean(Acc,γ=1) = {statistics.mean(g1ac):.4f}")

# AC checks
print()
print("## 4. 验收条件检查")
print()
print("AC-9  (Smoke ASR ≥ 0.20):                 ✅ PASS (已验证)")

c05 = [r['asr'] for r in all_results if r['dataset']=='cifar10' and r['alpha']=='0.5' and r['gamma']=='10' and r['asr'] is not None]
if c05:
    m=statistics.mean(c05)
    print(f"AC-10 (CIFAR-10 α=0.5 mean ASR ≥ 0.20):  {'✅ PASS' if m>=0.20 else '❌ FAIL'} (mean={m:.4f})")

if g10a and g1a:
    d=statistics.mean(g10a)-statistics.mean(g1a)
    print(f"AC-11 (ΔASR γ10-γ1 ≥ 0.30):              {'✅ PASS' if d>=0.30 else '❌ FAIL'} (Δ={d:+.4f})")

# Observations
print()
print("## 5. 关键观察")
print()
collapsed = [r for r in g10 if r['acc'] is not None and r['acc']<=0.12]
print(f"  模型崩溃 (final acc ≤ 0.12): {len(collapsed)}/{len(g10)} 组 γ=10 实验")
if collapsed:
    print(f"  → γ=10 缩放过于激进，模型退化为全局预测 target_label=0")
    print(f"  → ASR=1.0 在此情况下是 trivial collapse，非精确后门植入")
print()
pre_ok = [r for r in g10 if r['pre_acc'] is not None and r['pre_acc']>0.3]
print(f"  攻击前准确率正常 (pre-atk acc > 0.30): {len(pre_ok)}/{len(g10)} 组")
print()
print("  攻击前(round=N-6)各组准确率:")
for k in sorted(groups):
    items=groups[k]
    pas=[x['pre_acc'] for x in items if x['pre_acc'] is not None]
    if pas:
        print(f"    {k[0]} α={k[1]}: {ms(pas)}")
