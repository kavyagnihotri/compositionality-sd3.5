"""
Generate images from CogVLM Custom SD-3.5 and calculate FID score

Generates 1k images using COCO captions with CogVLM encoder.
Calculates FID against COCO ground-truth and appends to results.
"""

import sys
sys.path.append('..')
sys.path.append('../z4-cogvlm_custom_sd35')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics
from custom_sd35 import create_sd35_with_custom_encoder


# Configuration
CAPTIONS_CSV = "./coco_1k_captions.csv"
COCO_GT_FILE = "./coco_ground_truth.pt"
FID_RESULTS_CSV = "./fid_results.csv"
OUTPUT_DIR = "./generated_images"

CONVERTED_ENCODER_DIR = "../z4-cogvlm_custom_sd35/converted_text_encoder_CogVLM"

# Generation parameters
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 28
GUIDANCE = 3.5
SEED = 42

# FID parameters
BATCH_SIZE = 50

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TensorDataset(Dataset):
    """
    Dataset for pre-loaded tensors.
    torch-fidelity expects uint8 [0, 255] range, so we convert.
    """
    def __init__(self, tensors):
        """
        Args:
            tensors: torch.Tensor of shape [N, C, H, W] in range [0, 1]
        """
        # Convert to [0, 255] uint8 for torch-fidelity
        self.tensors = (tensors * 255).byte()
        print(f"  Loaded {len(self.tensors)} images, shape: {self.tensors.shape}")
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx]


def generate_images(pipe, captions_df, seed=42):
    """
    Generate images from prompts and return as stacked tensor.
    
    Args:
        pipe: Diffusion pipeline
        captions_df: DataFrame with 'caption' and 'image_id' columns
        seed: Random seed for generation
    
    Returns:
        dict with 'tensors' [N, C, H, W], 'captions', 'image_ids'
    """
    print(f"Generating images with CogVLM Custom SD-3.5")
    
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    all_tensors = []
    captions = []
    image_ids = []
    
    for idx, row in tqdm(captions_df.iterrows(), total=len(captions_df), desc="Generating (CogVLM)"):
        caption = row['caption']
        image_id = row['image_id']
        
        # Generate image
        with torch.no_grad():
            output = pipe(
                prompt=caption,
                num_inference_steps=NUM_STEPS,
                guidance_scale=GUIDANCE,
                height=HEIGHT,
                width=WIDTH,
                generator=generator,
            )
            image = output.images[0]
        
        # Convert PIL image to tensor [C, H, W] in range [0, 1]
        img_tensor = torch.from_numpy(
            np.array(image).transpose(2, 0, 1)  # HWC -> CHW
        ).float() / 255.0
        
        all_tensors.append(img_tensor)
        captions.append(caption)
        image_ids.append(image_id)
    
    # Stack all tensors into [N, C, H, W]
    stacked_tensors = torch.stack(all_tensors, dim=0)
    print(f"  Generated tensor shape: {stacked_tensors.shape}")
    
    return {
        'tensors': stacked_tensors,
        'captions': captions,
        'image_ids': image_ids,
        'num_images': len(captions),
        'height': HEIGHT,
        'width': WIDTH,
    }


def calculate_fid_from_tensors(tensor_data, coco_gt_file):
    """
    Calculate FID score between generated images and COCO ground-truth.
    
    Args:
        tensor_data: Dict with 'tensors' key containing [N, C, H, W] tensor
        coco_gt_file: Path to COCO ground-truth tensor file (.pt)
    
    Returns:
        FID score (float)
    """
    print(f"\nCalculating FID: CogVLM SD-3.5 vs COCO Ground-Truth")
    print(f"  COCO GT file: {coco_gt_file}")
    
    # Load COCO ground-truth
    coco_data = torch.load(coco_gt_file)
    coco_tensors = coco_data['tensors']
    
    # Extract generated tensors
    generated_tensors = tensor_data['tensors']
    
    # Create datasets
    generated_dataset = TensorDataset(generated_tensors)
    coco_dataset = TensorDataset(coco_tensors)
    
    if len(generated_dataset) == 0 or len(coco_dataset) == 0:
        print(f"  Error: Empty dataset(s)")
        return None
    
    # Calculate FID using torch-fidelity
    print(f"  Computing FID with batch_size={BATCH_SIZE}...")
    
    metrics = calculate_metrics(
        input1=generated_dataset,
        input2=coco_dataset,
        cuda=(DEVICE == "cuda"),
        batch_size=BATCH_SIZE,
        fid=True,
        verbose=False
    )
    
    fid_score = metrics['frechet_inception_distance']
    print(f"  FID Score: {fid_score:.4f}")
    
    return fid_score


def append_to_fid_results(fid_score, num_images, fid_results_csv):
    """
    Append CogVLM FID result to existing results CSV.
    
    Args:
        fid_score: FID score to append
        num_images: Number of images used
        fid_results_csv: Path to CSV file
    """
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    new_row = {
        'comparison': 'CogVLM_vs_COCO',
        'dataset_1': 'CogVLM Custom SD-3.5',
        'dataset_2': 'COCO Ground-Truth',
        'fid_score': fid_score,
        'num_images': num_images,
        'timestamp': timestamp
    }
    
    # Load existing results
    fid_results_path = Path(fid_results_csv)
    if fid_results_path.exists():
        df_existing = pd.read_csv(fid_results_path)
        # Append new row
        df_new = pd.concat([df_existing, pd.DataFrame([new_row])], ignore_index=True)
    else:
        # Create new dataframe
        df_new = pd.DataFrame([new_row])
    
    # Save back to file
    df_new.to_csv(fid_results_path, index=False)
    print(f"\nAppended results to {fid_results_path}")
    print(f"  CogVLM FID: {fid_score:.4f}")


def main():
    print("=" * 70)
    print("CogVLM CUSTOM SD-3.5: IMAGE GENERATION AND FID CALCULATION")
    print("=" * 70)
    print(f"Device: {DEVICE}")
    print(f"Parameters: {HEIGHT}×{WIDTH}, {NUM_STEPS} steps, guidance={GUIDANCE}")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load captions
    captions_df = pd.read_csv(CAPTIONS_CSV)
    print(f"\nLoaded {len(captions_df)} captions from {CAPTIONS_CSV}")
    
    # Check if COCO ground-truth exists
    coco_gt_file = Path(COCO_GT_FILE)
    if not coco_gt_file.exists():
        print(f"\nError: COCO ground-truth not found at {coco_gt_file}")
        print("Please run 1_download_coco_groundtruth.py first")
        return
    
    # ============ Load CogVLM Custom SD-3.5 Pipeline ============
    print("\n" + "=" * 70)
    print("STEP 1: LOADING COGVLM CUSTOM SD-3.5 PIPELINE")
    print("=" * 70)
    
    cogvlm_pipe = create_sd35_with_custom_encoder(CONVERTED_ENCODER_DIR)
    
    # ============ Generate Images ============
    print("\n" + "=" * 70)
    print("STEP 2: GENERATING IMAGES")
    print("=" * 70)
    
    cogvlm_data = generate_images(
        pipe=cogvlm_pipe,
        captions_df=captions_df,
        seed=SEED
    )
    
    # Save generated tensors (in same directory as standard and custom)
    tensor_path = output_path / "cogvlm_sd35.pt"
    torch.save(cogvlm_data, tensor_path)
    print(f"\nSaved {cogvlm_data['num_images']} images to {tensor_path}")
    
    # Clear GPU memory before FID calculation
    del cogvlm_pipe
    torch.cuda.empty_cache()
    
    # ============ Calculate FID ============
    print("\n" + "=" * 70)
    print("STEP 3: CALCULATING FID SCORE")
    print("=" * 70)
    
    fid_score = calculate_fid_from_tensors(
        tensor_data=cogvlm_data,
        coco_gt_file=coco_gt_file
    )
    
    if fid_score is None:
        print("\nError: FID calculation failed")
        return
    
    # ============ Append to Results ============
    print("\n" + "=" * 70)
    print("STEP 4: SAVING RESULTS")
    print("=" * 70)
    
    append_to_fid_results(
        fid_score=fid_score,
        num_images=cogvlm_data['num_images'],
        fid_results_csv=FID_RESULTS_CSV
    )
    
    # ============ Summary ============
    print("\n" + "=" * 70)
    print("GENERATION AND FID CALCULATION COMPLETE")
    print("=" * 70)
    print(f"Model:        CogVLM Custom SD-3.5")
    print(f"Images:       {cogvlm_data['num_images']}")
    print(f"Tensor File:  {tensor_path}")
    print(f"FID Score:    {fid_score:.4f}")
    print(f"Results CSV:  {FID_RESULTS_CSV}")
    print("=" * 70)


if __name__ == "__main__":
    main()
