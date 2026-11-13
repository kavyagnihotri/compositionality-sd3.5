import os
import argparse
import torch
import pandas as pd
from tqdm import tqdm
from dotenv import load_dotenv
from huggingface_hub import login
from torchvision.transforms import ToTensor
from diffusers import StableDiffusion3Pipeline

# Load environment variables
load_dotenv()

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--metadata_csv", required=True, help="Path to metadata CSV")
    parser.add_argument("--output_dir", required=True, help="Output directory")
    parser.add_argument("--batch_size", type=int, default=1, help="Images per prompt")
    return parser.parse_args()

def main():
    args = parse_args()
    
    # Verify HF token exists
    if not os.getenv("HF_TOKEN"):
        raise ValueError("HF_TOKEN not found in .env file")
    
    # Initialize pipeline
    print("Loading SD3.5 pipeline...")
    login(token=os.getenv("HF_TOKEN"))
    pipeline = StableDiffusion3Pipeline.from_pretrained("stabilityai/stable-diffusion-3.5-medium", torch_dtype=torch.bfloat16).to("cuda")

    # Load metadata
    df = pd.read_csv(args.metadata_csv)
    
    # Process each prompt
    results = []
    for _, row in tqdm(df.iterrows(), total=len(df)):
        try:
            # Create output directories
            img_dir = os.path.join(args.output_dir, row['category'], row['prompt_type'])
            os.makedirs(img_dir, exist_ok=True)
            
            # Generate images
            images = pipeline(
                row['prompt_text'],
                num_images_per_prompt=args.batch_size,
                generator=torch.Generator(device="cuda").manual_seed(42),
                output_type="pil"
            ).images
            
            # Save images
            for i, img in enumerate(images):
                png_path = os.path.join(img_dir, f"{row['image_index']}_{i}.png")
                tensor_path = os.path.join(img_dir, f"{row['image_index']}_{i}.pt")
                
                img.save(png_path)
                torch.save(ToTensor()(img), tensor_path)
                
                results.append({
                    'prompt': row['prompt_text'],
                    'category': row['category'],
                    'png_path': png_path,
                    'tensor_path': tensor_path
                })
                
        except Exception as e:
            print(f"Error with prompt '{row['prompt_text']}': {e}")
            continue
    
    # Save results
    pd.DataFrame(results).to_csv(os.path.join(args.output_dir, "results.csv"), index=False)
    print(f"Done! Generated {len(results)} images")

if __name__ == "__main__":
    main()
