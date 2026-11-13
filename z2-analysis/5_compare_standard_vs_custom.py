"""Compare Standard vs Custom SD3.5 using QWEN reason evaluations."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

BASELINE_QWEN_DIR = Path("../x7-z1-qwen_small/qwen_outputs_standard_sd/reason")
CUSTOM_QWEN_DIR = Path("../x7-z1-qwen_small/qwen_outputs")
OUTPUT_DIR = Path("./analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_baseline_results():
    """Load baseline SD3.5 QWEN evaluations."""
    results = {}
    for csv_file in BASELINE_QWEN_DIR.glob("*_qwen_reason_*.csv"):
        phrase = csv_file.stem.split("_qwen_reason_")[0]
        results[phrase] = pd.read_csv(csv_file)
    return results

def load_custom_results():
    """Load custom SD3.5 QWEN evaluations."""
    results = {}
    
    if not CUSTOM_QWEN_DIR.exists():
        return results
    
    for csv_file in CUSTOM_QWEN_DIR.glob("*_custom_qwen_reason_*.csv"):
        phrase = csv_file.stem.split("_custom_qwen_reason_")[0]
        df = pd.read_csv(csv_file)
        
        if phrase in results:
            results[phrase] = pd.concat([results[phrase], df], ignore_index=True)
        else:
            results[phrase] = df
    
    return results

def calc_yes_rate(df):
    """Calculate YES rate."""
    if len(df) == 0:
        return None
    return (df['answer'].str.upper() == 'YES').sum() / len(df) * 100

def count_responses(df):
    """Count YES, NO, and other responses."""
    if len(df) == 0:
        return {'yes': 0, 'no': 0, 'other': 0}
    
    yes_count = int((df['answer'].str.upper() == 'YES').sum())
    no_count = int((df['answer'].str.upper() == 'NO').sum())
    other_count = int(len(df) - yes_count - no_count)
    
    return {'yes': yes_count, 'no': no_count, 'other': other_count}

def main():
    baseline_results = load_baseline_results()
    custom_results = load_custom_results()
    
    overlapping = set(baseline_results.keys()) & set(custom_results.keys())
    
    comparison = {}
    for phrase in overlapping:
        baseline_yes = calc_yes_rate(baseline_results[phrase])
        custom_yes = calc_yes_rate(custom_results[phrase])
        baseline_counts = count_responses(baseline_results[phrase])
        custom_counts = count_responses(custom_results[phrase])
        
        comparison[phrase] = {
            'baseline': baseline_yes,
            'custom': custom_yes,
            'diff': custom_yes - baseline_yes,
            'baseline_n': len(baseline_results[phrase]),
            'custom_n': len(custom_results[phrase]),
            'baseline_yes': baseline_counts['yes'],
            'baseline_no': baseline_counts['no'],
            'baseline_other': baseline_counts['other'],
            'custom_yes': custom_counts['yes'],
            'custom_no': custom_counts['no'],
            'custom_other': custom_counts['other']
        }
    
    print(f"{'Phrase':<20} {'Baseline':>12} {'Custom':>12} {'Diff':>12} {'B(Y/N/O)':>12} {'C(Y/N/O)':>12}")
    for phrase in sorted(comparison.keys()):
        c = comparison[phrase]
        baseline_counts_str = f"{c['baseline_yes']}/{c['baseline_no']}/{c['baseline_other']}"
        custom_counts_str = f"{c['custom_yes']}/{c['custom_no']}/{c['custom_other']}"
        print(f"{phrase:<20} {c['baseline']:>11.1f}% {c['custom']:>11.1f}% {c['diff']:>+11.1f}% {baseline_counts_str:>12} {custom_counts_str:>12}")
    
    avg_baseline = np.mean([c['baseline'] for c in comparison.values()])
    avg_custom = np.mean([c['custom'] for c in comparison.values()])
    avg_diff = avg_custom - avg_baseline
    
    print(f"\nAverage: Baseline={avg_baseline:.1f}%, Custom={avg_custom:.1f}%, Diff={avg_diff:+.1f}%")
    
    better_overall = 'Custom' if avg_diff > 0 else 'Baseline'
    print(f"Better overall: {better_overall}")
    
    pd.DataFrame([{'phrase': p, **c} for p, c in comparison.items()]).to_csv(
        OUTPUT_DIR / "5_standard_vs_custom.csv", index=False)
    
    with open(OUTPUT_DIR / "5_standard_vs_custom.json", 'w') as f:
        json.dump({'comparison': comparison, 'avg_baseline': avg_baseline,
                   'avg_custom': avg_custom, 'better': better_overall}, f, indent=2)

if __name__ == "__main__":
    main()
