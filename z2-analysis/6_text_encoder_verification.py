"""Text Encoder Verification Summary."""

import pandas as pd
from pathlib import Path

VERIFICATION_DIR = Path("../z0.5-verification_checks")
OUTPUT_DIR = Path("./analysis_results")
OUTPUT_DIR.mkdir(exist_ok=True)

def main():
    csv_path = VERIFICATION_DIR / "conversion_verification.csv"
    
    if not csv_path.exists():
        print(f"File not found: {csv_path}")
        return
    
    df = pd.read_csv(csv_path)
    
    # Check column names match CSV format
    max_norm = df['Frobenius Norm'].max()
    mean_norm = df['Frobenius Norm'].mean()
    
    # Handle missing Max Absolute Diff values
    if 'Max Absolute Diff' in df.columns and not df['Max Absolute Diff'].isna().all():
        max_abs_diff = df['Max Absolute Diff'].max()
    else:
        max_abs_diff = max_norm  # Use Frobenius norm if Max Absolute Diff not available
    
    print(f"CLIC Text Encoder Conversion Verification")
    print(f"Rows checked: {len(df)}")
    print(f"Max Frobenius Norm: {max_norm:.6e}")
    print(f"Mean Frobenius Norm: {mean_norm:.6e}")
    print(f"Perfect conversion: {'Yes' if max_norm == 0.0 else 'No'}")
    
    summary = pd.DataFrame([{
        'rows_checked': len(df),
        'max_frobenius_norm': max_norm,
        'mean_frobenius_norm': mean_norm,
        'perfect_conversion': max_norm == 0.0
    }])
    
    summary.to_csv(OUTPUT_DIR / "6_text_encoder_verification.csv", index=False)
    print(f"\nSaved to {OUTPUT_DIR / '6_text_encoder_verification.csv'}")

if __name__ == "__main__":
    main()
