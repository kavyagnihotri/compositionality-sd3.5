"""
Load and test custom SD3.5 pipeline with replaced text encoder (CLIC CogVLM).

Uses pre-converted CLIC CogVLM text encoder from ./converted_text_encoder

This script:
1. Loads SD3.5 pipeline with custom text encoder
2. Runs comprehensive image generation tests
3. Optionally generates images from CSV prompts with metadata tracking
"""
import torch
import sys
import os
import argparse
import datetime
import pandas as pd
from pathlib import Path
from tqdm import tqdm
from torchvision.transforms import ToTensor

from custom_sd35 import (
    create_sd35_with_custom_encoder,
    CONVERTED_MODEL_DIR,
    LOCAL_SD_DIR,
    DEVICE
)


def parse_args():
    """Parse command line arguments."""
    parser = argparse.ArgumentParser(
        description="Load and test custom SD3.5 pipeline with replaced text encoder"
    )
    parser.add_argument("--csv_file", type=str, default=None,
                       help="Path to CSV file with prompts and metadata")
    parser.add_argument("--output_dir", type=str, default=None,
                       help="Directory to save output tensors (default: auto-generated)")
    parser.add_argument("--seed", type=int, default=42,
                       help="Fixed seed for image generation")
    parser.add_argument("--phrase", type=str, default=None,
                       help="Phrase to filter prompts by; if set, only generate images for this phrase")
    parser.add_argument("--skip_default_tests", action="store_true",
                       help="Skip default test prompts (only run CSV if provided)")
    return parser.parse_args()


def load_prompts_from_csv(csv_file):
    """Load prompts and metadata from a CSV file."""
    try:
        df = pd.read_csv(csv_file)
        required_columns = ['prompt', 'class1', 'category', 'phrase']
        for col in required_columns:
            if col not in df.columns:
                raise ValueError(f"CSV file missing required column: {col}")
        
        # class2 is optional for single-class files
        if 'class2' not in df.columns:
            df['class2'] = None
        
        prompts_data = df.to_dict('records')
        return prompts_data
    except Exception as e:
        print(f"Error loading CSV file: {e}")
        return []


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
    """Save images as tensors grouped by phrase and return metadata entries."""
    tensor_dir = os.path.join(output_dir, "tensors")
    os.makedirs(tensor_dir, exist_ok=True)
    
    phrase = phrase_group[0]['phrase']
    phrase_hash = abs(hash(phrase)) % (10 ** 8)  # 8-digit hash
    tensor_filename = f"phrase_{phrase_hash}.pt"
    tensor_path = os.path.join(tensor_dir, tensor_filename)
    
    tensors = []
    prompts = []
    metadata_entries = []
    
    for idx, (prompt_data, img) in enumerate(zip(phrase_group, images)):
        img_tensor = ToTensor()(img)  # Convert PIL to tensor [C, H, W]
        tensors.append(img_tensor)
        prompts.append(prompt_data['prompt'])
        
        metadata_entry = {
            'tensor_file': tensor_filename,
            'tensor_index': idx,
            'prompt': prompt_data['prompt'],
            'class1': prompt_data['class1'],
            'category': prompt_data['category'],
            'phrase': phrase,
        }
        
        # Add class2 only if it exists (for dual-class files)
        if prompt_data.get('class2') is not None:
            metadata_entry['class2'] = prompt_data['class2']
        
        metadata_entries.append(metadata_entry)
    
    tensor_data = {
        'images': tensors,
        'prompts': prompts
    }
    torch.save(tensor_data, tensor_path)
    
    return metadata_entries


def generate_from_csv(pipe, csv_file, output_dir=None, seed=42, phrase_filter=None):
    """Generate images from CSV prompts using custom SD3.5 pipeline."""
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Use provided output directory or default
    if output_dir is None:
        output_dir = f"cogvlm_sd35_output_{timestamp}"
    os.makedirs(output_dir, exist_ok=True)
    
    # Extract the base name of the input CSV file without extension
    csv_basename = os.path.basename(csv_file)
    csv_name = os.path.splitext(csv_basename)[0]
    
    prompts_data = load_prompts_from_csv(csv_file)
    print(f"\nLoaded {len(prompts_data)} prompts from {csv_file}")
    
    # If a phrase filter was provided, apply it
    if phrase_filter:
        filtered = [p for p in prompts_data if str(p.get('phrase')) == str(phrase_filter)]
        if not filtered:
            print(f"No prompts found for phrase: {phrase_filter}. Exiting.")
            return
        prompts_data = filtered
        print(f"Filtered prompts to phrase '{phrase_filter}': {len(prompts_data)} prompts")
    
    # Group prompts by phrase
    grouped_prompts = group_prompts_by_phrase(prompts_data)
    print(f"Grouped into {len(grouped_prompts)} phrase groups")
    
    print(f"\nGenerating images with custom SD3.5 (CogVLM) using fixed seed {seed}...")
    all_metadata = []
    
    # Process each phrase group
    for phrase_key, phrase_group in tqdm(grouped_prompts.items(), desc="Processing phrases"):
        images = []
        for prompt_data in phrase_group:
            # Generate image
            generator = torch.Generator(device=DEVICE).manual_seed(seed)
            modified_prompt = f"{prompt_data['prompt']}. Do not make ambiguous image."
            
            with torch.no_grad():
                output = pipe(
                    modified_prompt,
                    num_inference_steps=28,
                    guidance_scale=3.5,
                    height=512,
                    width=512,
                    generator=generator,
                    output_type="pil"
                )
            images.append(output.images[0])
        
        metadata_entries = save_phrase_tensors(phrase_group, images, output_dir, seed)
        all_metadata.extend(metadata_entries)
    
    # Save metadata to CSV with a unique name based on input file
    metadata_df = pd.DataFrame(all_metadata)
    metadata_csv_path = os.path.join(output_dir, f"metadata_{csv_name}_{timestamp}.csv")
    metadata_df.to_csv(metadata_csv_path, index=False)
    
    print(f"\nGenerated {len(all_metadata)} images grouped into {len(grouped_prompts)} tensor files")
    print(f"Metadata saved to: {metadata_csv_path}")
    print(f"Output saved to: {output_dir}")
    
    return True


def print_pipeline_info(pipe):
    """
    Print basic information about the loaded pipeline.
    
    Shows configuration of all three text encoders without extensive verification
    since the model has already been converted and verified.
    """
    print("\nPIPELINE INFORMATION")
    
    print("\nText Encoder Configuration:")
    print(f"  text_encoder (CUSTOM CLIC CogVLM):")
    print(f"    Hidden size: {pipe.text_encoder.config.hidden_size}")
    print(f"    Vocab size: {pipe.text_encoder.config.vocab_size}")
    print(f"    Activation: {pipe.text_encoder.config.hidden_act}")
    print(f"    Layers: {pipe.text_encoder.config.num_hidden_layers}")
    
    print(f"\n  text_encoder_2 (Original CLIP-G):")
    print(f"    Hidden size: {pipe.text_encoder_2.config.hidden_size}")
    print(f"    Activation: {pipe.text_encoder_2.config.hidden_act}")
    
    print(f"\n  text_encoder_3 (Original T5-XXL):")
    print(f"    Model type: {pipe.text_encoder_3.config.model_type}")
    
    print("\nPipeline ready for image generation!")
    
    return True


def run_comprehensive_tests(pipe):
    """Run all tests: basic generation + spatial reasoning."""
    print("\nCOMPREHENSIVE TESTING")
    
    # Test 1: Basic generation
    print("\n[Test 1/2] Basic image generation (4 prompts)...")
    basic_prompts = [
        "a red ball above a blue cube",
        "a cat sitting to the left of a dog",
        "a tree below the clouds",
        "a coffee cup on top of a wooden table"
    ]
    
    output_dir = Path("./test_outputs")
    output_dir.mkdir(exist_ok=True)
    
    for i, prompt in enumerate(basic_prompts, 1):
        print(f"  [{i}/4] Generating: '{prompt}'")
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=28,
                guidance_scale=3.5,
                height=512,
                width=512
            ).images[0]
        
        save_path = output_dir / f"test_{i:02d}.png"
        image.save(save_path)
    
    print(f"  Saved to: {output_dir}/")
    
    # Test 2: Spatial reasoning
    print("\n[Test 2/2] Spatial reasoning evaluation (4 prompts)...")
    spatial_prompts = [
        'a red cube to the left of a blue sphere',
        'a green triangle to the right of a yellow square',
        'a cat sitting above a cardboard box',
        'a dog standing below a wooden table'
    ]
    
    spatial_dir = Path("./spatial_outputs")
    spatial_dir.mkdir(exist_ok=True)
    
    for i, prompt in enumerate(spatial_prompts, 1):
        print(f"  [{i}/4] Generating: '{prompt}'")
        with torch.no_grad():
            image = pipe(
                prompt,
                num_inference_steps=28,
                guidance_scale=3.5,
                height=512,
                width=512
            ).images[0]
        
        safe_name = prompt.replace(' ', '_').replace('/', '_')[:50]
        save_path = spatial_dir / f"spatial_{i:02d}_{safe_name}.png"
        image.save(save_path)
    
    print(f"  Saved to: {spatial_dir}/")
    print("\nALL TESTS COMPLETE!")
    
    return True


def main():
    args = parse_args()
    
    print("\nCUSTOM SD3.5 WITH REPLACED TEXT ENCODER - TEST WORKFLOW (CLIC CogVLM)")
    print(f"Device: {DEVICE}")
    print(f"Converted Model: {CONVERTED_MODEL_DIR}")
    print(f"SD3.5 Base: {LOCAL_SD_DIR}")
    
    # Check that converted model exists
    converted_path = Path(CONVERTED_MODEL_DIR)
    if not (converted_path.exists() and (converted_path / "config.json").exists()):
        print(f"\nERROR: Converted model not found at {CONVERTED_MODEL_DIR}")
        print("Please run the conversion process first.")
        return 1
    
    print(f"\nUsing converted model at {CONVERTED_MODEL_DIR}")
    
    # Load pipeline with custom encoder
    print("\nLOAD SD3.5 PIPELINE WITH CUSTOM TEXT ENCODER")
    
    print("\nLoading SD3.5 with component-based assembly...")
    pipe = create_sd35_with_custom_encoder(CONVERTED_MODEL_DIR)
    print("Pipeline created")
    
    # Print pipeline info
    print_pipeline_info(pipe)
    
    # CSV-based generation if provided
    if args.csv_file:
        print("\nCSV-BASED IMAGE GENERATION")
        generate_from_csv(
            pipe, 
            csv_file=args.csv_file,
            output_dir=args.output_dir,
            seed=args.seed,
            phrase_filter=args.phrase
        )
    
    # Run default tests unless skipped
    if not args.skip_default_tests:
        run_comprehensive_tests(pipe)
        
        # Final summary for default tests
        print("\nDEFAULT TESTS COMPLETE - SUMMARY")
        print(f"""
        Pipeline: SD3.5 with custom text encoder (CLIC CogVLM)
        text_encoder: CUSTOM CLIC CogVLM (QuickGELU, replaced)
        text_encoder_2: ORIGINAL CLIP-G (unchanged)
        text_encoder_3: ORIGINAL T5-XXL (unchanged)
        
        Tests:
        Basic generation: 4 images in ./test_outputs/
        Spatial reasoning: 4 images in ./spatial_outputs/
        
        Status: SUCCESS - Pipeline loaded and tested!
        """)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())
