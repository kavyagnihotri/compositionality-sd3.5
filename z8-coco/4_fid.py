"""
Calculate FID scores for Step 3 outputs using pytorch-fid.

Loads generated image folders from step 3 and computes FID
against COCO ground-truth images.
"""

import sys
sys.path.append('..')

import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from pytorch_fid import fid_score


# Configuration
COCO_GT_DIR = "./coco_ground_truth_images"
FID_RESULTS_CSV = "./fid_results.csv"
OUTPUT_DIR = "./generated_images"

MODEL_IMAGE_DIRS = {
    "Standard SD-3.5": "standard_sd35_images",
    "CLIC SD-3.5": "clic_sd35_images",
    "CogVLM SD-3.5": "cogvlm_sd35_images",
}

# FID parameters
BATCH_SIZE = 50
FID_DIMS = 2048
NUM_WORKERS = 4

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def count_png_images(directory):
    """Count PNG files in a directory."""
    return len(list(Path(directory).glob("*.png")))


def calculate_fid_from_dirs(generated_images_dir, coco_gt_dir, model_name):
    """
    Calculate FID score between generated images and COCO ground-truth.
    
    Args:
        generated_images_dir: Directory containing generated images
        coco_gt_dir: Directory containing resized COCO ground-truth images
    
    Returns:
        FID score (float)
    """
    print(f"\nCalculating FID: {model_name} vs COCO Ground-Truth (pytorch-fid)")
    print(f"  Generated dir: {generated_images_dir}")
    print(f"  COCO GT dir:   {coco_gt_dir}")

    gen_dir = Path(generated_images_dir)
    gt_dir = Path(coco_gt_dir)
    if not gen_dir.exists() or not gt_dir.exists():
        print("  Error: Missing generated or ground-truth directory")
        return None

    print(f"  Computing FID with batch_size={BATCH_SIZE}, dims={FID_DIMS}...")

    fid_value = fid_score.calculate_fid_given_paths(
        paths=[str(gen_dir), str(gt_dir)],
        batch_size=BATCH_SIZE,
        device=DEVICE,
        dims=FID_DIMS,
        num_workers=NUM_WORKERS,
    )

    fid_score_value = float(fid_value)
    print(f"  FID Score: {fid_score_value:.4f}")

    return fid_score_value


def append_to_fid_results(rows, fid_results_csv):
    """
    Append FID results to existing results CSV.
    
    Args:
        rows: List of row dicts to append
        fid_results_csv: Path to CSV file
    """
    if not rows:
        return
    
    # Load existing results
    fid_results_path = Path(fid_results_csv)
    if fid_results_path.exists():
        df_existing = pd.read_csv(fid_results_path)
        df_new = pd.concat([df_existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        df_new = pd.DataFrame(rows)
    
    # Save back to file
    df_new.to_csv(fid_results_path, index=False)
    print(f"\nAppended {len(rows)} row(s) to {fid_results_path}")


def main():
    print("=" * 70)
    print("STEP 4: FID CALCULATION ONLY (NO IMAGE GENERATION)")
    print("=" * 70)
    print(f"Device: {DEVICE}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    # Check if COCO ground-truth image directory exists
    coco_gt_dir = Path(COCO_GT_DIR)
    if not coco_gt_dir.exists():
        print(f"\nError: COCO ground-truth directory not found at {coco_gt_dir}")
        print("Please run 2_coco_ground_truth.py first")
        return

    # ============ Calculate FID for each generated model ============
    print("\n" + "=" * 70)
    print("CALCULATING FID SCORES")
    print("=" * 70)

    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    result_rows = []

    for model_name, image_dir_name in MODEL_IMAGE_DIRS.items():
        image_dir = output_path / image_dir_name
        if not image_dir.exists():
            print(f"Skipping {model_name}: missing {image_dir}")
            continue

        num_images = count_png_images(image_dir)
        if num_images == 0:
            print(f"Skipping {model_name}: no PNG images in {image_dir}")
            continue

        fid_value = calculate_fid_from_dirs(
            generated_images_dir=image_dir,
            coco_gt_dir=coco_gt_dir,
            model_name=model_name,
        )
        if fid_value is None:
            print(f"  Failed FID for {model_name}")
            continue

        comparison_name = model_name.replace(" ", "_").replace("-", "") + "_vs_COCO"
        result_rows.append(
            {
                "comparison": comparison_name,
                "dataset_1": model_name,
                "dataset_2": "COCO Ground-Truth",
                "fid_score": fid_value,
                "num_images": num_images,
                "timestamp": timestamp,
            }
        )

    if not result_rows:
        print("\nError: No FID results were produced")
        return

    # ============ Append to Results CSV ============
    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    append_to_fid_results(result_rows, FID_RESULTS_CSV)

    # ============ Summary ============
    print("\n" + "=" * 70)
    print("FID CALCULATION COMPLETE")
    print("=" * 70)
    for row in result_rows:
        print(f"{row['dataset_1']:<20} FID: {row['fid_score']:.4f} ({row['num_images']} images)")
    print(f"Results CSV:  {FID_RESULTS_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
