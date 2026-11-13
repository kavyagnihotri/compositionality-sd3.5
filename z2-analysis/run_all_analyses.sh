#!/bin/bash


# Make sure to activate your Python environment first
source /usr/etc/profile.d/conda.sh
conda activate sd2

# Run all analysis scripts
# echo "1. Evaluator Comparison (ground truth validation)"
# python3 1_evaluator_comparison.py
# echo ""

# echo "1.5 Evaluator LLAVA Comparison (ground truth validation)"
# python3 1.5_llava_mode_comparison.py
# echo ""

# echo "2. QWEN Mode Comparison (ground truth validation)"
# python3 2_qwen_mode_comparison.py
# echo ""

echo "3. Baseline SD3.5 Evaluation Rates"
python3 3_baseline_sd35_qwen_reason.py
echo ""

echo "4. Custom SD3.5 Evaluation Rates"
python3 4_custom_sd35_qwen_reason.py
echo ""

echo "5. Standard vs Custom Comparison"
python3 5_compare_standard_vs_custom.py
echo ""

# echo "6. Text Encoder Verification"
# python3 6_text_encoder_verification.py
# echo ""

# echo "All analyses complete!"
