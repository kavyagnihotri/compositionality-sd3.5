"""Custom SD3.5 QWEN Reason evaluation rates."""

import pandas as pd
import numpy as np
from pathlib import Path
import json

CUSTOM_QWEN_DIR = Path("../x7-z1-qwen_small/qwen_outputs")
OUTPUT_DIR = Path("./analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def load_custom_qwen_results():
    """Load custom SD3.5 QWEN evaluations."""
    results = {}
    
    if not CUSTOM_QWEN_DIR.exists():
        return results
    
    csv_files = list(CUSTOM_QWEN_DIR.glob("*_custom_qwen_reason_*.csv"))
    
    for csv_file in csv_files:
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
    results = load_custom_qwen_results()
    
    phrase_stats = {}
    for phrase, df in results.items():
        counts = count_responses(df)
        phrase_stats[phrase] = {
            'yes_rate': calc_yes_rate(df),
            'n': len(df),
            'yes_count': counts['yes'],
            'no_count': counts['no'],
            'other_count': counts['other']
        }
    
    print(f"{'Phrase':<20} {'YES Rate':>12} {'YES':>6} {'NO':>6} {'Other':>6} {'Total':>6}")
    for phrase in sorted(phrase_stats.keys()):
        s = phrase_stats[phrase]
        print(f"{phrase:<20} {s['yes_rate']:>11.1f}% {s['yes_count']:>6} {s['no_count']:>6} {s['other_count']:>6} {s['n']:>6}")
    
    print(f"\nLowest YES rates:")
    for phrase, s in sorted(phrase_stats.items(), key=lambda x: x[1]['yes_rate'])[:5]:
        print(f"  {phrase}: {s['yes_rate']:.1f}%")
    
    print(f"\nHighest YES rates:")
    for phrase, s in sorted(phrase_stats.items(), key=lambda x: x[1]['yes_rate'], reverse=True)[:5]:
        print(f"  {phrase}: {s['yes_rate']:.1f}%")
    
    pd.DataFrame([{'phrase': p, **s} for p, s in phrase_stats.items()]).to_csv(
        OUTPUT_DIR / "4_custom_sd35_qwen_reason.csv", index=False)
    
    with open(OUTPUT_DIR / "4_custom_sd35_qwen_reason.json", 'w') as f:
        json.dump(phrase_stats, f, indent=2)

if __name__ == "__main__":
    main()
