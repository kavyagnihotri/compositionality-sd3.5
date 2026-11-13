"""
Calculate FID scores for Step 3 outputs.

Loads generated image folders from step 3 and computes FID against raw COCO
ground-truth images using either pytorch-fid or clean-fid.
"""

import argparse
from datetime import datetime
from pathlib import Path

import pandas as pd
import torch
from PIL import Image
from pytorch_fid import fid_score as pytorch_fid_score
from tqdm import tqdm

try:
    from cleanfid import fid as cleanfid_fid
except ImportError:  # pragma: no cover - runtime dependency message only
    cleanfid_fid = None


# Configuration
COCO_GT_DIR = "./coco_ground_truth_images_raw"
COCO_GT_DIR_RESIZED = "./coco_ground_truth_images_resized"
FID_RESULTS_CSV = "./fid_results.csv"
OUTPUT_DIR = "./generated_images"

# Image dimensions (must match generation parameters)
HEIGHT = 512
WIDTH = 512

MODEL_IMAGE_DIRS = {
    "Standard SD-3.5": "standard_sd35_images",
    "CLIC SD-3.5": "clic_sd35_images",
    "CogVLM SD-3.5": "cogvlm_sd35_images",
}


def count_png_images(directory):
    """Count PNG files in a directory."""
    return len(list(Path(directory).glob("*.png")))


def count_jpg_images(directory):
    """Count JPG files in a directory."""
    return len(list(Path(directory).glob("*.jpg")))


def resize_coco_images_for_pytorch_fid(raw_coco_dir, resized_output_dir, height, width):
    """
    Resize raw COCO JPGs to match generation dimensions for pytorch-fid.
    
    Clean-FID handles resizing internally, so this is only needed for pytorch-fid.
    
    Args:
        raw_coco_dir: Directory containing raw COCO JPG images
        resized_output_dir: Output directory for resized images
        height: Target height
        width: Target width
    
    Returns:
        Number of images resized
    """
    raw_dir = Path(raw_coco_dir)
    out_dir = Path(resized_output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    jpg_images = list(raw_dir.glob("*.jpg"))
    if not jpg_images:
        print(f"  No JPG images found in {raw_dir}")
        return 0
    
    resized_count = 0
    print(f"  Resizing {len(jpg_images)} COCO images to {height}x{width}...")
    
    for jpg_path in tqdm(jpg_images, desc="Resizing"):
        try:
            image = Image.open(jpg_path).convert("RGB")
            image = image.resize((width, height), Image.LANCZOS)
            output_path = out_dir / jpg_path.name
            image.save(output_path)
            resized_count += 1
        except Exception as e:
            print(f"  Warning: Failed to resize {jpg_path.name}: {e}")
    
    return resized_count


def parse_args():
    parser = argparse.ArgumentParser(
        description="Calculate FID scores for generated images against COCO ground truth."
    )
    parser.add_argument(
        "--use-clean-fid",
        action="store_true",
        help="Use clean-fid instead of pytorch-fid.",
    )
    return parser.parse_args()


def calculate_fid_from_dirs(generated_images_dir, coco_gt_dir, model_name, use_clean_fid=False):
    """
    Calculate FID score between generated images and COCO ground-truth.
    
    For pytorch-fid: uses resized COCO images (512x512)
    For clean-fid: uses raw COCO images (arbitrary dimensions)
    """
    backend_name = "Clean-FID" if use_clean_fid else "pytorch-fid"
    print(f"\nCalculating {backend_name}: {model_name} vs COCO Ground-Truth")
    print(f"  Generated dir: {generated_images_dir}")
    print(f"  COCO GT dir:   {coco_gt_dir}")

    gen_dir = Path(generated_images_dir)
    gt_dir = Path(coco_gt_dir)
    if not gen_dir.exists() or not gt_dir.exists():
        print("  Error: Missing generated or ground-truth directory")
        return None

    if use_clean_fid:
        if cleanfid_fid is None:
            print("  Error: clean-fid is not installed. Install it with `pip install clean-fid`.")
            return None

        fid_score_value = cleanfid_fid.compute_fid(str(gen_dir), str(gt_dir))
    else:
        fid_score_value = pytorch_fid_score.calculate_fid_given_paths(
            paths=[str(gen_dir), str(gt_dir)],
            batch_size=50,
            device="cuda" if torch.cuda.is_available() else "cpu",
            dims=2048,
            num_workers=4,
        )

    print(f"  {backend_name} Score: {float(fid_score_value):.4f}")

    return float(fid_score_value)


def append_to_fid_results(rows, fid_results_csv):
    """
    Append FID results to existing results CSV.

    Args:
        rows: List of row dicts to append
        fid_results_csv: Path to CSV file
    """
    if not rows:
        return

    fid_results_path = Path(fid_results_csv)
    if fid_results_path.exists():
        df_existing = pd.read_csv(fid_results_path)
        df_new = pd.concat([df_existing, pd.DataFrame(rows)], ignore_index=True)
    else:
        df_new = pd.DataFrame(rows)

    df_new.to_csv(fid_results_path, index=False)
    print(f"\nAppended {len(rows)} row(s) to {fid_results_path}")


def main():
    args = parse_args()

    print("=" * 70)
    print("STEP 4: FID CALCULATION ONLY (NO IMAGE GENERATION)")
    print("=" * 70)

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    coco_gt_dir = Path(COCO_GT_DIR)
    if not coco_gt_dir.exists():
        print(f"\nError: COCO ground-truth directory not found at {coco_gt_dir}")
        print("Please run 2_coco_ground_truth.py first")
        return

    backend_name = "Clean-FID" if args.use_clean_fid else "pytorch-fid"
    
    # For pytorch-fid, resize COCO images to match generation dimensions
    if not args.use_clean_fid:
        print("\n" + "=" * 70)
        print("PREPARING RESIZED COCO IMAGES FOR PYTORCH-FID")
        print("=" * 70)
        
        coco_resized_dir = Path(COCO_GT_DIR_RESIZED)
        num_resized = resize_coco_images_for_pytorch_fid(coco_gt_dir, coco_resized_dir, HEIGHT, WIDTH)
        
        if num_resized == 0:
            print(f"\nError: Failed to resize COCO images")
            return
        
        print(f"\nResized {num_resized} COCO images to {HEIGHT}x{WIDTH}")
        coco_gt_dir = coco_resized_dir  # Use resized directory for pytorch-fid

    print("\n" + "=" * 70)
    print(f"CALCULATING {backend_name.upper()} SCORES")
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
            use_clean_fid=args.use_clean_fid,
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

    print("\n" + "=" * 70)
    print("SAVING RESULTS")
    print("=" * 70)

    append_to_fid_results(result_rows, FID_RESULTS_CSV)

    print("\n" + "=" * 70)
    print(f"{backend_name.upper()} CALCULATION COMPLETE")
    print("=" * 70)
    for row in result_rows:
        print(f"{row['dataset_1']:<20} {backend_name}: {row['fid_score']:.4f} ({row['num_images']} images)")
    print(f"Results CSV:  {FID_RESULTS_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
