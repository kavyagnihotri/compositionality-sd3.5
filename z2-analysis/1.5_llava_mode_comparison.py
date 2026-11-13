"""Compare LLaVA Normal vs Incontext modes against ground truth."""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from sklearn.metrics import accuracy_score, f1_score, cohen_kappa_score

LLAVA_DIR = Path("../x6-llava_vlm/llava_outputs")
MANUAL_ANNOTATIONS = Path("../x5.5-prompts_sd35_output/problematic_images.csv")
OUTPUT_DIR = Path("./analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PHRASES = ['above', 'below', 'to_the_left_of', 'to_the_right_of']

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

def load_llava_predictions(mode, phrase):
    """Load LLaVA predictions. mode: 'normal' or 'incontext'."""
    if mode == 'normal':
        pattern = f"{phrase}_llava_output_*.csv"
        # Exclude incontext files
        files = [f for f in LLAVA_DIR.glob(pattern) if 'incontext' not in f.name]
    else:
        pattern = f"{phrase}_incontext_llava_output_*.csv"
        files = list(LLAVA_DIR.glob(pattern))
    
    if not files:
        return None
    
    df = pd.read_csv(files[0])
    return {(row['tensor_file'], row['tensor_index']): 1 if row['answer'].upper() == 'YES' else 0
            for _, row in df.iterrows()}

def compare_modes(ground_truth):
    """Compare LLaVA normal vs incontext."""
    results = {}
    
    for phrase in PHRASES:
        normal_preds = load_llava_predictions('normal', phrase)
        incontext_preds = load_llava_predictions('incontext', phrase)
        
        if not normal_preds or not incontext_preds:
            continue
        
        y_true, y_normal, y_incontext = [], [], []
        
        for (p, tensor_file, img_type), sets in ground_truth.items():
            if p != phrase:
                continue
            
            ambiguous_set = sets['ambiguous']
            incorrect_set = sets['incorrect']
            
            # Check all predictions that exist in both modes
            for pred_key in normal_preds.keys():
                if pred_key[0] == tensor_file and pred_key in incontext_preds:
                    idx = pred_key[1]
                    if idx not in ambiguous_set:
                        true_label = 0 if idx in incorrect_set else 1
                        y_true.append(true_label)
                        y_normal.append(normal_preds[pred_key])
                        y_incontext.append(incontext_preds[pred_key])
        
        if y_true:
            results[phrase] = {
                'normal_acc': accuracy_score(y_true, y_normal),
                'incontext_acc': accuracy_score(y_true, y_incontext),
                'normal_f1': f1_score(y_true, y_normal, zero_division=0),
                'incontext_f1': f1_score(y_true, y_incontext, zero_division=0),
                'agreement': cohen_kappa_score(y_normal, y_incontext),
                'n': len(y_true)
            }
    
    return results

def main():
    ground_truth = load_ground_truth()
    results = compare_modes(ground_truth)
    
    print(f"{'Phrase':<20} {'Normal Acc':>12} {'InCtx Acc':>12} {'Normal F1':>12} {'InCtx F1':>12} {'Kappa':>10} {'N':>6}")
    for phrase, r in results.items():
        print(f"{phrase:<20} {r['normal_acc']:>12.3f} {r['incontext_acc']:>12.3f} {r['normal_f1']:>12.3f} {r['incontext_f1']:>12.3f} {r['agreement']:>10.3f} {r['n']:>6}")
    
    overall = {
        'normal_acc': np.mean([r['normal_acc'] for r in results.values()]),
        'incontext_acc': np.mean([r['incontext_acc'] for r in results.values()]),
        'normal_f1': np.mean([r['normal_f1'] for r in results.values()]),
        'incontext_f1': np.mean([r['incontext_f1'] for r in results.values()])
    }
    
    print(f"\nOverall: Normal Acc={overall['normal_acc']:.3f}, InContext Acc={overall['incontext_acc']:.3f}")
    print(f"         Normal F1={overall['normal_f1']:.3f}, InContext F1={overall['incontext_f1']:.3f}")
    
    better = 'incontext' if overall['incontext_f1'] > overall['normal_f1'] else 'normal'
    print(f"\nBetter mode: {better}")
    
    pd.DataFrame([{'phrase': p, **r} for p, r in results.items()]).to_csv(
        OUTPUT_DIR / "1.5_llava_mode_comparison.csv", index=False)
    
    with open(OUTPUT_DIR / "1.5_llava_mode_comparison.json", 'w') as f:
        json.dump({'results': results, 'overall': overall, 'better': better}, f, indent=2)

if __name__ == "__main__":
    main()
