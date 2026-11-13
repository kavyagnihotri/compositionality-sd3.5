"""
Prepare raw COCO ground-truth images for Clean-FID evaluation.

Downloads COCO 2017 validation images corresponding to the sampled prompts
and copies the raw JPG files into a local directory without resizing.
"""

import shutil
from pathlib import Path
from urllib.request import urlretrieve
import zipfile

import pandas as pd
from tqdm import tqdm


# Configuration
CAPTIONS_CSV = "./coco_5k_captions.csv"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_DIR = "./coco_val2017"
OUTPUT_DIR = "./coco_ground_truth_images_raw"


def download_coco_images():
    """Download COCO 2017 validation images."""
    coco_dir = Path(COCO_VAL_DIR)
    zip_path = coco_dir.parent / "val2017.zip"

    # COCO zip extracts to a folder named "val2017"
    extracted_dir = coco_dir.parent / "val2017"

    if extracted_dir.exists() and len(list(extracted_dir.glob("*.jpg"))) > 0:
        print(f"COCO images already exist at {extracted_dir}")
        return extracted_dir

    coco_dir.mkdir(parents=True, exist_ok=True)

    # Download
    urlretrieve(COCO_VAL_IMAGES_URL, zip_path)
    print(f"Downloaded to {zip_path}")

    # Extract
    with zipfile.ZipFile(zip_path, "r") as zip_ref:
        zip_ref.extractall(coco_dir.parent)
    print(f"Extracted to {coco_dir.parent}")

    # Cleanup
    zip_path.unlink()

    return extracted_dir


def write_ground_truth_images(captions_df, coco_images_dir, output_dir):
    """
    Copy selected COCO images to output directory without resizing.

    Clean-FID performs any required preprocessing internally, so the raw JPGs
    are preserved as-is.
    """
    print("WRITING RAW COCO GROUND-TRUTH IMAGES")
    coco_dir = Path(coco_images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)

    saved = 0
    missing = []

    for idx, row in tqdm(captions_df.iterrows(), total=len(captions_df), desc="Copying"):
        image_id = row["image_id"]

        # COCO filename format: 000000123456.jpg (12 digits)
        coco_filename = f"{image_id:012d}.jpg"
        coco_path = coco_dir / coco_filename

        if not coco_path.exists():
            missing.append(image_id)
            continue

        output_name = f"{idx:05d}_{int(image_id):012d}.jpg"
        shutil.copy2(coco_path, out_dir / output_name)
        saved += 1

    print(f"\nSaved {saved} raw ground-truth images")
    print(f"  Output dir: {out_dir}")
    if missing:
        print(f"  Missing {len(missing)} images: {missing}...")

    return saved


def main():
    print("PREPARE RAW COCO GROUND-TRUTH IMAGE DIRECTORY")

    if not Path(CAPTIONS_CSV).exists():
        print(f"Error: {CAPTIONS_CSV} not found")
        return

    captions_df = pd.read_csv(CAPTIONS_CSV)
    print(f"Loaded {len(captions_df)} captions")

    coco_dir = download_coco_images()
    num_saved = write_ground_truth_images(captions_df, coco_dir, OUTPUT_DIR)

    print("COMPLETE")
    print(f"Ground-truth images saved: {num_saved}")
    print(f"Directory: {OUTPUT_DIR}")


if __name__ == "__main__":
    main()
