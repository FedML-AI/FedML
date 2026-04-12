import json
from pathlib import Path

import numpy as np

RESULTS_DIR = Path(__file__).resolve().parents[1] / 'n50_results' / 'metrics'

def load_exp(model, dataset, alpha, seed):
    path = RESULTS_DIR / f'metrics_{model}_{dataset}_shieldfl_atknone_defnone_a{alpha}_pmr0.0_gauto_seed{seed}.jsonl'
    if not path.exists():
        return None
    with path.open(encoding='utf-8') as f:
        data = [json.loads(l) for l in f]
    return data

print("=" * 80)
print("COMPREHENSIVE RESULTS EXTRACTION")
print("=" * 80)

for ds_label, model, dataset in [("CIFAR-10 + ResNet-18", "ResNet18", "cifar10"),
                                   ("MNIST + LeNet-5", "LeNet5", "mnist")]:
    print(f"\n{'='*60}")
    print(f"  {ds_label}")
    print(f"{'='*60}")
    
    for alpha in ["0.1", "0.5", "100"]:
        print(f"\n  alpha = {alpha}")
        all_final = []
        all_last10_mean = []
        nan_count = 0
        
        for seed in range(5):
            data = load_exp(model, dataset, alpha, seed)
            if data is None:
                print(f"    seed={seed}: MISSING!")
                continue
            
            accs = [d['test_accuracy'] for d in data]
            losses = [d['test_loss'] for d in data]
            n_rounds = len(data)
            
            has_nan = any(np.isnan(a) for a in accs) or any(np.isnan(l) for l in losses)
            if has_nan:
                nan_count += 1
            
            final_acc = accs[-1]
            max_acc = max(accs)
            max_round = accs.index(max_acc)
            last10 = accs[-10:]
            l10_mean = np.mean(last10)
            l10_std = np.std(last10)
            
            if n_rounds >= 20:
                trend = np.mean(accs[-10:]) - np.mean(accs[-20:-10])
            else:
                trend = 0
            
            all_final.append(final_acc)
            all_last10_mean.append(l10_mean)
            
            print(f"    seed={seed}: final={final_acc*100:.2f}%  max={max_acc*100:.2f}%@R{max_round}"
                  f"  last10={l10_mean*100:.2f}%+/-{l10_std*100:.2f}%"
                  f"  trend={trend*100:+.2f}pp  rounds={n_rounds}  NaN={'YES' if has_nan else 'no'}")
        
        if all_final:
            mean_final = np.mean(all_final) * 100
            std_final = np.std(all_final) * 100
            mean_l10 = np.mean(all_last10_mean) * 100
            print(f"    -- SUMMARY: final={mean_final:.2f}%+/-{std_final:.2f}%"
                  f"  last10_avg={mean_l10:.2f}%  NaN_seeds={nan_count}/5")

print("\n")
print("=" * 80)
print("AC-0-3: SEED DETERMINISM CHECK (C6 = CIFAR-10 a=0.5 seed=0)")
print("=" * 80)

c6_data = load_exp("ResNet18", "cifar10", "0.5", 0)
if c6_data:
    c6_final = c6_data[-1]['test_accuracy'] * 100
    anchor = 73.31
    delta = abs(c6_final - anchor)
    print(f"  C6 final acc: {c6_final:.2f}%")
    print(f"  Anchor (SA control): {anchor:.2f}%")
    print(f"  Delta: {delta:.2f}pp")
    print(f"  AC-0-3 (< 0.5pp): {'PASS' if delta < 0.5 else 'FAIL'}")

print("\n")
print("=" * 80)
print("AC-2: STATISTICAL CONSISTENCY")
print("=" * 80)

for ds_label, model, dataset in [("CIFAR-10", "ResNet18", "cifar10"),
                                   ("MNIST", "LeNet5", "mnist")]:
    means = {}
    for alpha in ["0.1", "0.5", "100"]:
        finals = []
        for seed in range(5):
            data = load_exp(model, dataset, alpha, seed)
            if data:
                finals.append(data[-1]['test_accuracy'])
        means[alpha] = np.mean(finals) if finals else 0
    
    mono = means["100"] > means["0.5"] > means["0.1"]
    print(f"  AC-2-1 {ds_label}: a=100({means['100']*100:.2f}%) > a=0.5({means['0.5']*100:.2f}%) > a=0.1({means['0.1']*100:.2f}%): {'PASS' if mono else 'FAIL'}")

for alpha in ["0.1", "0.5", "100"]:
    c10_finals = [load_exp("ResNet18", "cifar10", alpha, s)[-1]['test_accuracy'] for s in range(5)]
    mn_finals = [load_exp("LeNet5", "mnist", alpha, s)[-1]['test_accuracy'] for s in range(5)]
    c10_mean = np.mean(c10_finals)
    mn_mean = np.mean(mn_finals)
    ok = mn_mean > c10_mean
    print(f"  AC-2-2 a={alpha}: MNIST({mn_mean*100:.2f}%) > CIFAR-10({c10_mean*100:.2f}%): {'PASS' if ok else 'FAIL'}")

print()
for ds_label, model, dataset in [("CIFAR-10", "ResNet18", "cifar10"),
                                   ("MNIST", "LeNet5", "mnist")]:
    for alpha in ["0.1", "0.5", "100"]:
        finals = [load_exp(model, dataset, alpha, s)[-1]['test_accuracy'] for s in range(5)]
        std_val = np.std(finals) * 100
        if dataset == "cifar10":
            threshold = 3.0 if alpha != "0.1" else 999
        else:
            threshold = 2.0
        
        if alpha == "0.1" and dataset == "cifar10":
            status = "N/A (extreme non-IID)"
        else:
            status = f"{'PASS' if std_val < threshold else 'FAIL'} (threshold={threshold}pp)"
        print(f"  AC-2-3 {ds_label} a={alpha}: std={std_val:.2f}pp  {status}")

print()
for ds_label, model, dataset in [("CIFAR-10", "ResNet18", "cifar10"),
                                   ("MNIST", "LeNet5", "mnist")]:
    for alpha in ["0.1", "0.5", "100"]:
        for seed in range(5):
            data = load_exp(model, dataset, alpha, seed)
            accs = [d['test_accuracy'] for d in data]
            last10_std = np.std(accs[-10:]) * 100
            status = "PASS" if last10_std < 1.0 else ("WARN" if alpha == "0.1" and dataset == "cifar10" else "FAIL")
            if last10_std >= 1.0:
                print(f"  AC-2-4 {ds_label} a={alpha} s={seed}: last10_std={last10_std:.2f}pp  {status}")

print("\n")
print("=" * 80)
print("AC-1: ACCURACY RANGE CHECK")
print("=" * 80)

ac1_ranges = {
    ("cifar10", "0.1"): (35, 65),
    ("cifar10", "0.5"): (65, 80),
    ("cifar10", "100"): (73, 85),
    ("mnist", "0.1"): (90, 98.5),
    ("mnist", "0.5"): (94, 99.5),
    ("mnist", "100"): (96, 99.5),
}

for ds_label, model, dataset in [("CIFAR-10", "ResNet18", "cifar10"),
                                   ("MNIST", "LeNet5", "mnist")]:
    for alpha in ["0.1", "0.5", "100"]:
        finals = [load_exp(model, dataset, alpha, s)[-1]['test_accuracy'] * 100 for s in range(5)]
        mean_val = np.mean(finals)
        lo, hi = ac1_ranges[(dataset, alpha)]
        in_range = lo <= mean_val <= hi
        print(f"  AC-1 {ds_label} a={alpha}: mean={mean_val:.2f}%  range=[{lo}%, {hi}%]  {'PASS' if in_range else 'FAIL'}")
        if not in_range:
            print(f"    ! Out of range by {max(0, lo-mean_val, mean_val-hi):.2f}pp")
