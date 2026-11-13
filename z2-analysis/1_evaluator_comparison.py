"""
Compare VLM Evaluators vs Ground Truth

Compares LLaVA, QWEN simple, and QWEN reason against manual annotations.
Analyzes 5 phrases: above, below, to_the_left_of, to_the_right_of, between
"""

import pandas as pd
import numpy as np
from pathlib import Path
import json
import re
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

LLAVA_DIR = Path("../x6-llava_vlm/llava_outputs")
QWEN_SIMPLE_DIR = Path("../x7-z1-qwen_small/qwen_outputs_standard_sd/simple")
QWEN_REASON_DIR = Path("../x7-z1-qwen_small/qwen_outputs_standard_sd/reason")
MANUAL_ANNOTATIONS = Path("../x5.5-prompts_sd35_output/problematic_images.csv")
OUTPUT_DIR = Path("./analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

PHRASES = ['above', 'below', 'to_the_left_of', 'to_the_right_of', 'between']

def load_ground_truth():
    """Load manual annotations. Indices match tensor_index in predictions."""
    ground_truth = {}
    ambiguous_counts = {}
    
    with open(MANUAL_ANNOTATIONS, 'r') as f:
        lines = f.read().strip().split('\n')[1:]  # Skip header
    
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
        
        # Extract indices from [ambiguous],[incorrect] format
        bracket_matches = re.findall(r'\[([^\]]*)\]', ','.join(parts[3:]))
        if len(bracket_matches) < 2:
            continue
        
        # Parse semicolon-separated indices (as-is, could be 0-99 or 1-100)
        ambiguous = set([int(x.strip()) for x in bracket_matches[0].split(';') if x.strip() and x.strip().isdigit()])
        incorrect = set([int(x.strip()) for x in bracket_matches[1].split(';') if x.strip() and x.strip().isdigit()])
        
        # Store ambiguous and incorrect sets (will match against actual predictions)
        key = (phrase, tensor_file, img_type)
        ground_truth[key] = {'ambiguous': ambiguous, 'incorrect': incorrect}
        ambiguous_counts[key] = len(ambiguous)
    
    return ground_truth, ambiguous_counts

def load_evaluator_predictions(evaluator, phrase):
    """Load predictions from an evaluator."""
    dirs = {'llava': LLAVA_DIR, 'qwen_simple': QWEN_SIMPLE_DIR, 'qwen_reason': QWEN_REASON_DIR}
    patterns = {'llava': f"{phrase}_*llava_output*.csv", 
                'qwen_simple': f"{phrase}_qwen_output*.csv",
                'qwen_reason': f"{phrase}_qwen_reason*.csv"}
    
    files = list(dirs[evaluator].glob(patterns[evaluator]))
    if not files:
        return None
    
    df = pd.read_csv(files[0])
    return {(row['tensor_file'], row['tensor_index']): 
            1 if row['answer'].upper() == 'YES' else 0 
            for _, row in df.iterrows()}

def evaluate(evaluator, ground_truth, ambiguous_counts):
    """Evaluate an evaluator against ground truth."""
    results = {}
    
    for phrase in PHRASES:
        predictions = load_evaluator_predictions(evaluator, phrase)
        if not predictions:
            continue
        
        y_true, y_pred, total_ambiguous = [], [], 0
        
        for (p, tensor_file, img_type), sets in ground_truth.items():
            if p != phrase:
                continue
            total_ambiguous += ambiguous_counts.get((p, tensor_file, img_type), 0)
            
            ambiguous_set = sets['ambiguous']
            incorrect_set = sets['incorrect']
            
            # Check all predictions, skip ambiguous, label as 0 (incorrect) or 1 (correct)
            for pred_key, pred_value in predictions.items():
                if pred_key[0] == tensor_file:
                    idx = pred_key[1]
                    if idx not in ambiguous_set:
                        true_label = 0 if idx in incorrect_set else 1
                        y_true.append(true_label)
                        y_pred.append(pred_value)
        
        if y_true:
            results[phrase] = {
                'accuracy': accuracy_score(y_true, y_pred),
                'precision': precision_score(y_true, y_pred, zero_division=0),
                'recall': recall_score(y_true, y_pred, zero_division=0),
                'f1': f1_score(y_true, y_pred, zero_division=0),
                'n': len(y_true),
                'ambiguous': total_ambiguous
            }
    
    return results

def main():
    ground_truth, ambiguous_counts = load_ground_truth()
    print(f"Loaded {len(PHRASES)} phrases, {sum(ambiguous_counts.values())} ambiguous images excluded\n")
    
    evaluators = ['llava', 'qwen_simple', 'qwen_reason']
    all_results = {e: evaluate(e, ground_truth, ambiguous_counts) for e in evaluators}
    
    # Print results
    print(f"{'Phrase':<20} {'Evaluator':<15} {'Acc':>8} {'Prec':>8} {'Rec':>8} {'F1':>8} {'N':>6} {'Amb':>6}")
    for phrase in PHRASES:
        for evaluator in evaluators:
            if phrase in all_results[evaluator]:
                r = all_results[evaluator][phrase]
                print(f"{phrase:<20} {evaluator:<15} {r['accuracy']:>8.3f} {r['precision']:>8.3f} "
                      f"{r['recall']:>8.3f} {r['f1']:>8.3f} {r['n']:>6} {r['ambiguous']:>6}")
    
    # Overall scores
    overall = {e: {'accuracy': np.mean([m['accuracy'] for m in all_results[e].values()]),
                   'f1': np.mean([m['f1'] for m in all_results[e].values()])}
               for e in evaluators if all_results[e]}
    
    print(f"\n{'Evaluator':<15} {'Mean Acc':>12} {'Mean F1':>12}")
    for e, scores in overall.items():
        print(f"{e:<15} {scores['accuracy']:>12.3f} {scores['f1']:>12.3f}")
    
    best = max(overall.items(), key=lambda x: x[1]['f1'])[0]
    print(f"\nBest: {best}")
    
    # Save results
    pd.DataFrame([{'phrase': p, 'evaluator': e, **all_results[e][p]} 
                  for e in evaluators for p in PHRASES if p in all_results[e]]).to_csv(
        OUTPUT_DIR / "1_evaluator_comparison.csv", index=False)
    
    with open(OUTPUT_DIR / "1_evaluator_comparison.json", 'w') as f:
        json.dump({'results': all_results, 'overall': overall, 'best': best}, f, indent=2)

if __name__ == "__main__":
    main()
