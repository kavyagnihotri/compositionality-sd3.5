"""
Prepare COCO ground-truth images for FID evaluation.

Downloads COCO 2017 validation images corresponding to the sampled prompts
and writes resized images to a local directory for pytorch-fid.
"""

import pandas as pd
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import zipfile


# Configuration
CAPTIONS_CSV = "./coco_5k_captions.csv"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_DIR = "./coco_val2017"
OUTPUT_DIR = "./coco_ground_truth_images"

# Match generation parameters
HEIGHT = 512
WIDTH = 512


def download_coco_images():
    """Download COCO 2017 validation images."""
    coco_dir = Path(COCO_VAL_DIR)
    zip_path = coco_dir.parent / "val2017.zip"
    
    # COCO zip extracts to a folder named "val2017"
    extracted_dir = coco_dir.parent / "val2017"
    
    if extracted_dir.exists() and len(list(extracted_dir.glob('*.jpg'))) > 0:
        print(f"COCO images already exist at {extracted_dir}")
        return extracted_dir
    
    coco_dir.mkdir(parents=True, exist_ok=True)
    
    # Download
    urlretrieve(COCO_VAL_IMAGES_URL, zip_path)
    print(f"Downloaded to {zip_path}")
    
    # Extract
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(coco_dir.parent)
    print(f"Extracted to {coco_dir.parent}")
    
    # Cleanup
    zip_path.unlink()
    
    return extracted_dir


def write_ground_truth_images(captions_df, coco_images_dir, output_dir):
    """
    Write selected COCO images to output directory.
    Resizes to 512x512 for consistent FID evaluation.
    """
    print("WRITING COCO GROUND-TRUTH IMAGES")
    coco_dir = Path(coco_images_dir)
    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    
    saved = 0
    missing = []
    
    for idx, row in tqdm(captions_df.iterrows(), total=len(captions_df), desc="Converting"):
        image_id = row['image_id']
        caption = row['caption']
        
        # COCO filename format: 000000123456.jpg (12 digits)
        coco_filename = f"{image_id:012d}.jpg"
        coco_path = coco_dir / coco_filename
        
        if not coco_path.exists():
            missing.append(image_id)
            continue
        
        # Keep deterministic filenames so generated/gt sets align by index.
        image = Image.open(coco_path).convert('RGB')
        image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)

        output_name = f"{idx:05d}_{int(image_id):012d}.png"
        image.save(out_dir / output_name)
        saved += 1

    print(f"\nSaved {saved} resized ground-truth images")
    print(f"  Output dir: {out_dir}")
    if missing:
        print(f"  Missing {len(missing)} images: {missing}...")

    return saved


def main():
    print("PREPARE COCO GROUND-TRUTH IMAGE DIRECTORY")
    # Load captions
    if not Path(CAPTIONS_CSV).exists():
        print(f"Error: {CAPTIONS_CSV} not found")
        return
    
    captions_df = pd.read_csv(CAPTIONS_CSV)
    print(f"Loaded {len(captions_df)} captions")
    
    # Download COCO images
    coco_dir = download_coco_images()
    
    # Write resized images for pytorch-fid
    num_saved = write_ground_truth_images(captions_df, coco_dir, OUTPUT_DIR)
    
    print("COMPLETE")
    print(f"Ground-truth images saved: {num_saved}")
    print(f"Directory: {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
