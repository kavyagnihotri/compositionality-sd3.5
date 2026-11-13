"""
Generate images from Standard SD-3.5 and Custom SD-3.5

Generates 1k images from each model using COCO captions.
Saves as tensors only (for FID calculation with torch-fidelity).
"""

import sys
sys.path.append('..')

import torch
import pandas as pd
import numpy as np
from pathlib import Path
from tqdm import tqdm
from datetime import datetime
from diffusers import StableDiffusion3Pipeline
from custom_sd35 import create_sd35_with_custom_encoder


# Configuration
CAPTIONS_CSV = "./coco_1k_captions.csv"
OUTPUT_DIR = "./generated_images"

STANDARD_SD35_DIR = "../stable-diffusion-3.5-large-turbo"
CUSTOM_ENCODER_DIR = "../converted_text_encoder"

# Generation parameters
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 28
GUIDANCE = 3.5
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def generate_images(pipe, captions_df, model_name, seed=42):
    """
    Generate images from prompts and return as stacked tensor.
    
    Args:
        pipe: Diffusion pipeline
        captions_df: DataFrame with 'caption' and 'image_id' columns
        model_name: Name for logging (e.g., "Standard SD-3.5")
        seed: Random seed for generation
    
    Returns:
        dict with 'tensors' [N, C, H, W], 'captions', 'image_ids'
    """
    print(f"Generating images with {model_name}")
    
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    
    all_tensors = []
    captions = []
    image_ids = []
    
    for idx, row in tqdm(captions_df.iterrows(), total=len(captions_df), desc=f"Generating ({model_name})"):
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


def main():
    print("Image Generation:")
    print(f"Device: {DEVICE}")
    print(f"Parameters: {HEIGHT}×{WIDTH}, {NUM_STEPS} steps, guidance={GUIDANCE}")
    
    # Create output directory
    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)
    
    # Load captions
    captions_df = pd.read_csv(CAPTIONS_CSV)
    print(f"Loaded {len(captions_df)} captions")
    
    # Save prompts reference
    prompts_path = output_path / "prompts.csv"
    captions_df.to_csv(prompts_path, index=False)
    print(f"Saved prompts to {prompts_path}")
    
    # ============ Generate with Standard SD-3.5 ============
    print("LOADING STANDARD SD-3.5")
    
    standard_pipe = StableDiffusion3Pipeline.from_pretrained(
        STANDARD_SD35_DIR,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    standard_data = generate_images(
        pipe=standard_pipe,
        captions_df=captions_df,
        model_name="Standard SD-3.5",
        seed=SEED
    )
    
    # Save single tensor file
    standard_tensor_path = output_path / "standard_sd35.pt"
    torch.save(standard_data, standard_tensor_path)
    print(f"Saved {standard_data['num_images']} images to {standard_tensor_path}")
    
    # Clear GPU memory
    del standard_pipe
    torch.cuda.empty_cache()
    
    # ============ Generate with Custom SD-3.5 ============
    print("LOADING CUSTOM SD-3.5")
    
    custom_pipe = create_sd35_with_custom_encoder(CUSTOM_ENCODER_DIR)
    
    custom_data = generate_images(
        pipe=custom_pipe,
        captions_df=captions_df,
        model_name="Custom SD-3.5",
        seed=SEED
    )
    
    # Save single tensor file
    custom_tensor_path = output_path / "custom_sd35.pt"
    torch.save(custom_data, custom_tensor_path)
    print(f"Saved {custom_data['num_images']} images to {custom_tensor_path}")
    
    # Summary
    print("GENERATION COMPLETE")
    print(f"Standard SD-3.5: {standard_data['tensors'].shape}")
    print(f"  File: {standard_tensor_path}")
    print(f"Custom SD-3.5: {custom_data['tensors'].shape}")
    print(f"  File: {custom_tensor_path}")
    print(f"\nAll tensors generated and saved")

if __name__ == "__main__":
    main()
