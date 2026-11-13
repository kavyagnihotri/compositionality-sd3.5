# SD3.5 Spatial Reasoning Analysis

Analysis scripts for evaluating spatial reasoning in standard and custom SD3.5 models.

## Overview

This analysis compares:
- Standard SD3.5 (baseline) vs Custom SD3.5 (with CLIC text encoder)
- Different VLM evaluators: LLaVA, QWEN simple, QWEN reason
- Ground truth validation using manual annotations (5 phrases only)

## Ground Truth Availability

Manual annotations available for only 5 phrases:
- above
- below
- to_the_left_of
- to_the_right_of
- between

All other phrases report QWEN evaluations only (not validated against ground truth).

## Scripts

### 1. Evaluator Comparison (1_evaluator_comparison.py)

Compares LLaVA, QWEN simple, and QWEN reason against manual annotations.

Outputs:
- Accuracy, precision, recall, F1 for each evaluator
- Best evaluator recommendation

Usage:
```bash
cd analysis
python 1_evaluator_comparison.py
```

Output: `analysis_results/evaluator_comparison.csv`

### 2. QWEN Mode Comparison (2_qwen_mode_comparison.py)

Compares QWEN simple vs QWEN reason modes against ground truth.

Outputs:
- Performance metrics for each mode
- Agreement between modes (Cohen kappa)
- Which mode performs better

Usage:
```bash
python 2_qwen_mode_comparison.py
```

Output: `analysis_results/qwen_mode_comparison.csv`

### 3. Baseline SD3.5 Analysis (3_baseline_sd35_qwen_reason.py)

QWEN reason evaluation rates for standard SD3.5 across all spatial concepts.

Outputs:
- YES/NO rates for each phrase
- Marks which phrases have ground truth available
- Summary statistics

Usage:
```bash
python 3_baseline_sd35_qwen_reason.py
```

Output: `analysis_results/baseline_sd35_qwen_reason.csv`

### 4. Custom SD3.5 Analysis (4_custom_sd35_qwen_reason.py)

QWEN reason evaluation rates for custom SD3.5 (22 phrases).

Note: No ground truth available for custom SD3.5 images.

Outputs:
- YES/NO rates for each phrase
- Summary statistics

Usage:
```bash
python 4_custom_sd35_qwen_reason.py
```

Output: `analysis_results/custom_sd35_qwen_reason.csv`

### 5. Standard vs Custom Comparison (5_compare_standard_vs_custom.py)

Compares baseline vs custom SD3.5 using QWEN reason evaluations.

Outputs:
- YES rate changes for overlapping phrases
- Statistical significance (paired t-test, p<0.05)
- Improvements and degradations

Usage:
```bash
python 5_compare_standard_vs_custom.py
```

Output: `analysis_results/standard_vs_custom_comparison.csv`

### 6. Text Encoder Verification (6_text_encoder_verification.py)

Summarizes verification results for CLIC text encoder conversion.

Outputs:
- Conversion accuracy (should be perfect)
- Weight differences (confirms compositional training)
- Functional verification status

Usage:
```bash
python 6_text_encoder_verification.py
```

Output: `analysis_results/text_encoder_verification_summary.txt`

### Visualization Notebook (visualize_predictions.ipynb)

Interactive Jupyter notebook for visualizing images with QWEN predictions.

Features:
- Select model (baseline or custom)
- Browse by phrase
- Filter by QWEN answer (YES/NO)
- Shows image, prompt, and QWEN reasoning

Usage:
```bash
jupyter notebook visualize_predictions.ipynb
```

## Data Directories

Scripts expect these directories in parent folder:

```
../qwen_outputs/simple/          QWEN simple evaluations (baseline)
../qwen_outputs/reason/          QWEN reason evaluations (baseline)
../llava_outputs/                LLaVA evaluations
../problematic_images.csv        Manual annotations (ground truth)
../spatial_outputs/              Baseline SD3.5 generated images
../custom_sd35_output_*/         Custom SD3.5 generated images
../verification checks/          Text encoder verification results
```

## Output Directory

All results saved to: `../analysis_results/`

## Terminology

QWEN evaluation rates: What QWEN judged as correct/incorrect
- NOT the same as accuracy (which requires ground truth validation)
- Use scripts 1-2 for ground truth validated accuracy (5 phrases only)

YES rate: Percentage of images QWEN judged as having correct spatial relationship
NO rate: Percentage of images QWEN judged as having incorrect spatial relationship

## Running All Analyses

```bash
cd analysis

# Ground truth validation (5 phrases)
python 1_evaluator_comparison.py
python 2_qwen_mode_comparison.py

# Evaluation rates (all phrases)
python 3_baseline_sd35_qwen_reason.py
python 4_custom_sd35_qwen_reason.py

# Model comparison
python 5_compare_standard_vs_custom.py

# Text encoder verification
python 6_text_encoder_verification.py

# Visual inspection
jupyter notebook visualize_predictions.ipynb
```

## Notes

1. Only 5 phrases have ground truth annotations
2. Scripts 1-2 provide validated accuracy metrics
3. Scripts 3-4 provide QWEN evaluation rates (not validated)
4. Script 5 compares models using QWEN evaluations
5. Script 6 verifies text encoder conversion was successful
6. Visualization notebook allows manual spot-checking of predictions

## Dependencies

- pandas
- numpy
- scikit-learn
- scipy
- torch
- matplotlib
- PIL
- ipywidgets (for notebook)
