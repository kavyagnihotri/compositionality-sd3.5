"""
Download and sample 1000 COCO captions for FID evaluation

Downloads COCO 2017 validation annotations and samples 1k captions.
Saves to CSV for reproducibility.
"""

import json
import pandas as pd
import random
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


# Configuration
OUTPUT_CSV = "./coco_1k_captions.csv"
NUM_SAMPLES = 1000
SEED = 42

# COCO 2017 validation annotations URL
ANNOTATIONS_URL = "http://images.cocodataset.org/annotations/annotations_trainval2017.zip"


def download_coco_annotations():
    """Download COCO annotations if not already present."""
    annotations_dir = Path("./coco_annotations")
    annotations_file = annotations_dir / "annotations" / "captions_val2017.json"
    
    if annotations_file.exists():
        print(f"Annotations already exist at {annotations_file}")
        return annotations_file
    
    print("Downloading COCO 2017 annotations...")
    annotations_dir.mkdir(exist_ok=True)
    zip_path = annotations_dir / "annotations.zip"
    
    urlretrieve(ANNOTATIONS_URL, zip_path)
    print(f"Downloaded to {zip_path}")
    
    print("Extracting annotations...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extractall(annotations_dir)
    
    print(f"Extracted to {annotations_dir}")
    zip_path.unlink() 
    
    return annotations_file


def sample_captions(annotations_file, num_samples=1000, seed=42):
    """
    Sample N captions from COCO validation set.
    
    Returns: DataFrame with columns: image_id, caption
    """
    print(f"\nLoading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data['annotations']
    print(f"Total captions in COCO val: {len(annotations)}")
    
    random.seed(seed)
    sampled = random.sample(annotations, num_samples)
    
    data = [
        {
            'image_id': ann['image_id'],
            'caption': ann['caption']
        }
        for ann in sampled
    ]
    
    df = pd.DataFrame(data)
    print(f"\nSampled {len(df)} captions")
    print(f"Unique images: {df['image_id'].nunique()}")
    
    return df


def main():
    print("COCO CAPTION SAMPLING FOR FID EVALUATION")
    
    # Download annotations
    annotations_file = download_coco_annotations()
    
    # Sample captions
    df = sample_captions(annotations_file, num_samples=NUM_SAMPLES, seed=SEED)
    
    # Save to CSV
    output_path = Path(OUTPUT_CSV)
    df.to_csv(output_path, index=False)
    print(f"\nSaved {len(df)} captions to {output_path}")
    

if __name__ == "__main__":
    main()
