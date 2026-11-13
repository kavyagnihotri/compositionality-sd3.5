"""
Prepare COCO ground-truth tensors for FID evaluation

Downloads COCO 2017 validation images corresponding to the 1k prompts
and converts them to tensors for FID comparison.
"""

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from PIL import Image
from tqdm import tqdm
from urllib.request import urlretrieve
import zipfile
import os


# Configuration
CAPTIONS_CSV = "./coco_1k_captions.csv"
COCO_VAL_IMAGES_URL = "http://images.cocodataset.org/zips/val2017.zip"
COCO_VAL_DIR = "./coco_val2017"
OUTPUT_FILE = "./coco_ground_truth.pt"

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


def convert_coco_to_tensors(captions_df, coco_images_dir):
    """
    Convert COCO images to stacked tensor.
    Resizes to 512x512 and returns stacked tensors.
    """
    print("CONVERTING COCO IMAGES TO TENSORS")
    coco_dir = Path(coco_images_dir)
    
    all_tensors = []
    captions = []
    image_ids = []
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
        
        # Load and resize image
        image = Image.open(coco_path).convert('RGB')
        image = image.resize((WIDTH, HEIGHT), Image.LANCZOS)
        
        # Convert to tensor [C, H, W] in range [0, 1]
        img_array = np.array(image)
        img_tensor = torch.from_numpy(
            img_array.transpose(2, 0, 1)  # HWC -> CHW
        ).float() / 255.0
        
        all_tensors.append(img_tensor)
        captions.append(caption)
        image_ids.append(image_id)
    
    # Stack all tensors into [N, C, H, W]
    stacked_tensors = torch.stack(all_tensors, dim=0)
    
    print(f"\nConverted {len(all_tensors)} images")
    print(f"  Tensor shape: {stacked_tensors.shape}")
    if missing:
        print(f"  Missing {len(missing)} images: {missing}...")
    
    return {
        'tensors': stacked_tensors,
        'captions': captions,
        'image_ids': image_ids,
        'num_images': len(captions),
        'height': HEIGHT,
        'width': WIDTH,
    }


def main():
    print("PREPARE COCO GROUND-TRUTH TENSORS")
    # Load captions
    if not Path(CAPTIONS_CSV).exists():
        print(f"Error: {CAPTIONS_CSV} not found")
        return
    
    captions_df = pd.read_csv(CAPTIONS_CSV)
    print(f"Loaded {len(captions_df)} captions")
    
    # Download COCO images
    coco_dir = download_coco_images()
    
    # Convert to tensors and stack
    coco_data = convert_coco_to_tensors(captions_df, coco_dir)
    
    # Save single tensor file
    output_path = Path(OUTPUT_FILE)
    torch.save(coco_data, output_path)
    print(f"\nSaved {coco_data['num_images']} images to {output_path}")
    print(f"  File size: {output_path.stat().st_size / (1024**2):.2f} MB")
    
    print("COMPLETE")
    print(f"COCO ground-truth tensors: {coco_data['tensors'].shape}")
    print(f"File: {output_path}")

if __name__ == "__main__":
    main()
