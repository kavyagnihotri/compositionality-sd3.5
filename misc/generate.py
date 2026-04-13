import os
import torch
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
from torchvision.transforms import ToTensor

# Import custom SD3.5 pipeline
from custom_sd35_pipeline import create_custom_sd35_pipeline

# --------- CONFIG ----------
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

# Create output directory with timestamp
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--csv_file", type=str, required=True,
                       help="Path to CSV file with prompts and metadata")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save output tensors (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Fixed seed for image generation")
    parser.add_argument("--guidance_scale", type=float, default=0.0,
                       help="Guidance scale for image generation (default: 0.0)")
    return parser.parse_args()

def generate_images(pipeline, prompt, seed=42, guidance_scale=0.0):
    """Generate a single image from a prompt."""
    generator = torch.Generator(device=DEVICE).manual_seed(seed)
    output = pipeline(
        prompt,
        num_images_per_prompt=1,
        generator=generator,
        output_type="pil",
        guidance_scale=guidance_scale,
        num_inference_steps=4,
    )
    return output.images[0]  # Return single image

def group_prompts_by_phrase(prompts_data):
    """Group prompts by phrase to store in common tensors."""
    grouped = {}
    for data in prompts_data:
        phrase = data['phrase']
        if phrase not in grouped:
            grouped[phrase] = []
        grouped[phrase].append(data)
    return grouped

def save_phrase_tensors(phrase_group, images, output_dir, seed):
    """
    Save images for a phrase group as tensors and return metadata.
    
    Args:
        phrase_group: List of dictionaries containing prompt data for a phrase
        images: List of PIL images corresponding to the prompts
        output_dir: Root output directory
        seed: Seed used for generation
    
    Returns:
        List of dictionaries with metadata for each prompt/image
    """
    # Create tensor directory
    tensor_dir = os.path.join(output_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)
    
    # All images in this group share the same phrase
    phrase = phrase_group[0]['phrase']
    phrase_hash = abs(hash(phrase)) % (10 ** 8)  # 8-digit hash
    tensor_filename = f"phrase_{phrase_hash}.pt"
    tensor_path = os.path.join(tensor_dir, tensor_filename)
    
    # Convert all images to tensors
    tensors = []
    prompts = []
    metadata_entries = []
    
    for idx, (prompt_data, img) in enumerate(zip(phrase_group, images)):
        img_tensor = ToTensor()(img)  # Convert PIL to tensor [C, H, W]
        tensors.append(img_tensor)
        prompts.append(prompt_data['prompt'])
        
        # Create metadata entry for CSV
        metadata_entries.append({
            'tensor_file': tensor_filename,
            'tensor_index': idx,
            'prompt': prompt_data['prompt'],
            'class1': prompt_data['class1'],
            'class2': prompt_data['class2'],
            'category': prompt_data['category'],
            'phrase': phrase,
        })
    
    # Save all tensors for this phrase in one file
    tensor_data = {
        'images': tensors,
        'prompts': prompts
    }
    torch.save(tensor_data, tensor_path)
    
    return metadata_entries

def load_prompts_from_csv(csv_file):
    """Load prompts and metadata from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['prompt', 'class1', 'class2', 'category', 'phrase']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV file missing required column: {col}")
        
        # Convert DataFrame to list of dictionaries
        prompts_data = df.to_dict('records')
        return prompts_data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []

def run_experiment(csv_file, output_dir=None, seed=42, guidance_scale=0.0):
    # Use provided output directory or default
    if output_dir:
        os.makedirs(output_dir, exist_ok=True)
    else:
        output_dir = f"custom_sd35_output_{timestamp}"
        os.makedirs(output_dir, exist_ok=True)

    # Extract the base name of the input CSV file without extension
    csv_basename = os.path.basename(csv_file)
    csv_name = os.path.splitext(csv_basename)[0]
    
    prompts_data = load_prompts_from_csv(csv_file)
    print(f"Loaded {len(prompts_data)} prompts from {csv_file}")

    # Group prompts by phrase
    grouped_prompts = group_prompts_by_phrase(prompts_data)
    print(f"Grouped into {len(grouped_prompts)} phrase groups")
    
    # Setup custom SD3.5 pipeline with OpenCLIP
    pipeline = create_custom_sd35_pipeline(device=DEVICE)

    print(f"\nGenerating images with Custom SD3.5 using seed={seed}, guidance_scale={guidance_scale}...")
    all_metadata = []
    
    # Process each phrase group
    for phrase, phrase_group in tqdm(grouped_prompts.items(), desc="Processing phrase groups"):
        # Generate images for all prompts in this phrase group
        images = []
        for prompt_data in tqdm(phrase_group, desc=f"Generating '{phrase}'", leave=False):
            img = generate_images(pipeline, prompt_data['prompt'], seed=seed, guidance_scale=guidance_scale)
            images.append(img)
        
        # Save tensors and collect metadata
        metadata_entries = save_phrase_tensors(phrase_group, images, output_dir, seed)
        all_metadata.extend(metadata_entries)

    del pipeline
    torch.cuda.empty_cache()

    # Save metadata to CSV with a unique name based on input file
    metadata_df = pd.DataFrame(all_metadata)
    metadata_csv_path = os.path.join(output_dir, f"metadata_{csv_name}_{timestamp}.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)

    print(f"\n{'='*70}")
    print(f"COMPLETED!")
    print(f"Generated {len(all_metadata)} images grouped into {len(grouped_prompts)} tensor files")
    print(f"Metadata saved to: {metadata_csv_path}")
    print(f"Output saved to: {output_dir}")
    print(f"{'='*70}")

# Run the experiment
if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        csv_file=args.csv_file,
        output_dir=args.output_dir,
        seed=args.seed,
        guidance_scale=args.guidance_scale
    )

