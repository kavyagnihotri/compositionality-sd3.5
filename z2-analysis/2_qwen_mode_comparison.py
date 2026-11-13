"""Compare QWEN Simple vs Reason modes against ground truth."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

QWEN_SIMPLE_DIR = Path("../x7-z1-qwen_small/qwen_outputs_standard_sd/simple")
QWEN_REASON_DIR = Path("../x7-z1-qwen_small/qwen_outputs_standard_sd/reason")
MANUAL_ANNOTATIONS = Path("../x5.5-prompts_sd35_output/problematic_images.csv")
OUTPUT_DIR = Path("./analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PHRASES = ['above', 'below', 'to_the_left_of', 'to_the_right_of', 'between']

def load_ground_truth():
    """Load manual annotations. Indices match tensor_index in predictions."""
    ground_truth = {}
    
    with open(MANUAL_ANNOTATIONS, 'r') as f:
        lines = f.read().strip().split('\n')[1:]
    
    for line in lines:
        parts = line.split(',')
        if len(parts) < 5:
            continue
        
        phrase = parts[0].strip().replace(' ', '_')
        tensor_file = parts[1].strip()
        try:
            img_type = int(parts[2].strip())
        except:
            continue
        
        bracket_matches = re.findall(r'\[([^\]]*)\]', ','.join(parts[3:]))
        if len(bracket_matches) < 2:
            continue
        
        # Parse indices as-is (could be 0-99 or 1-100 depending on phrase)
        ambiguous = set([int(x.strip()) for x in bracket_matches[0].split(';') if x.strip() and x.strip().isdigit()])
        incorrect = set([int(x.strip()) for x in bracket_matches[1].split(';') if x.strip() and x.strip().isdigit()])
        
        ground_truth[(phrase, tensor_file, img_type)] = {'ambiguous': ambiguous, 'incorrect': incorrect}
    
    return ground_truth

def load_qwen_predictions(mode, phrase):
    """Load QWEN predictions."""
    dir_path = QWEN_SIMPLE_DIR if mode == 'simple' else QWEN_REASON_DIR
    pattern = f"{phrase}_qwen_output*.csv" if mode == 'simple' else f"{phrase}_qwen_reason*.csv"
    files = list(dir_path.glob(pattern))
    if not files:
        return None
    
    df = pd.read_csv(files[0])
    return {(row['tensor_file'], row['tensor_index']): 1 if row['answer'].upper() == 'YES' else 0
            for _, row in df.iterrows()}

def compare_modes(ground_truth):
    """Compare QWEN simple vs reason."""
    results = {}
    
    for phrase in PHRASES:
        simple_preds = load_qwen_predictions('simple', phrase)
        reason_preds = load_qwen_predictions('reason', phrase)
        
        if not simple_preds or not reason_preds:
            continue
        
        y_true, y_simple, y_reason = [], [], []
        
        for (p, tensor_file, img_type), sets in ground_truth.items():
            if p != phrase:
                continue
            
            ambiguous_set = sets['ambiguous']
            incorrect_set = sets['incorrect']
            
            # Check all predictions that exist in both modes
            for pred_key in simple_preds.keys():
                if pred_key[0] == tensor_file and pred_key in reason_preds:
                    idx = pred_key[1]
                    if idx not in ambiguous_set:
                        true_label = 0 if idx in incorrect_set else 1
                        y_true.append(true_label)
                        y_simple.append(simple_preds[pred_key])
                        y_reason.append(reason_preds[pred_key])
        
        if y_true:
            results[phrase] = {
                'simple_acc': accuracy_score(y_true, y_simple),
                'reason_acc': accuracy_score(y_true, y_reason), 
                'simple_f1': f1_score(y_true, y_simple, zero_division=0),
                'reason_f1': f1_score(y_true, y_reason, zero_division=0),
                'agreement': cohen_kappa_score(y_simple, y_reason),
                'n': len(y_true)
            }
    
    return results

def main():
    ground_truth = load_ground_truth()
    results = compare_modes(ground_truth)
    
    print(f"{'Phrase':<20} {'Simple Acc':>12} {'Reason Acc':>12} {'Simple F1':>12} {'Reason F1':>12} {'Kappa':>10} {'N':>6}")
    for phrase, r in results.items():
        print(f"{phrase:<20} {r['simple_acc']:>12.3f} {r['reason_acc']:>12.3f} {r['simple_f1']:>12.3f} {r['reason_f1']:>12.3f} {r['agreement']:>10.3f} {r['n']:>6}")
    
    overall = {
        'simple_acc': np.mean([r['simple_acc'] for r in results.values()]),
        'reason_acc': np.mean([r['reason_acc'] for r in results.values()]),
        'simple_f1': np.mean([r['simple_f1'] for r in results.values()]),
        'reason_f1': np.mean([r['reason_f1'] for r in results.values()])
    }
    
    print(f"\nOverall: Simple Acc={overall['simple_acc']:.3f}, Reason Acc={overall['reason_acc']:.3f}")
    print(f"         Simple F1={overall['simple_f1']:.3f}, Reason F1={overall['reason_f1']:.3f}")
    
    better = 'reason' if overall['reason_f1'] > overall['simple_f1'] else 'simple'
    print(f"\nBetter mode: {better}")
    
    pd.DataFrame([{'phrase': p, **r} for p, r in results.items()]).to_csv(
        OUTPUT_DIR / "2_qwen_mode_comparison.csv", index=False)
    
    with open(OUTPUT_DIR / "2_qwen_mode_comparison.json", 'w') as f:
        json.dump({'results': results, 'overall': overall, 'better': better}, f, indent=2)

if __name__ == "__main__":
    main()
