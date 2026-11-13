"""
Download and sample 5000 COCO captions for FID evaluation.

Downloads COCO 2017 validation annotations and samples 5k unique-image captions.
Saves to CSV for reproducibility.
"""

import json
import pandas as pd
import random
from pathlib import Path
from urllib.request import urlretrieve
import zipfile


# Configuration
OUTPUT_CSV = "./coco_5k_captions.csv"
NUM_SAMPLES = 5000
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


def sample_captions(annotations_file, num_samples=5000, seed=42):
    """
    Sample N captions from COCO validation set with unique image IDs.
    
    Returns: DataFrame with columns: image_id, caption
    """
    print(f"\nLoading annotations from {annotations_file}...")
    with open(annotations_file, 'r') as f:
        coco_data = json.load(f)
    
    annotations = coco_data['annotations']
    print(f"Total captions in COCO val: {len(annotations)}")

    # Group captions by image_id so each sampled row maps to a unique image.
    captions_by_image = {}
    for ann in annotations:
        image_id = ann['image_id']
        if image_id not in captions_by_image:
            captions_by_image[image_id] = []
        captions_by_image[image_id].append(ann['caption'])

    unique_image_ids = list(captions_by_image.keys())
    print(f"Unique images available: {len(unique_image_ids)}")

    if num_samples > len(unique_image_ids):
        raise ValueError(
            f"Requested {num_samples} samples, but only {len(unique_image_ids)} unique images are available"
        )

    rng = random.Random(seed)
    sampled_image_ids = rng.sample(unique_image_ids, num_samples)

    data = []
    for image_id in sampled_image_ids:
        caption = rng.choice(captions_by_image[image_id])
        data.append({'image_id': image_id, 'caption': caption})
    
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
