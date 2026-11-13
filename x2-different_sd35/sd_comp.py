import os
import torch
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from torchvision.transforms import ToTensor
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Login to Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--samples_per_category", type=int, default=7,
                       help="Number of samples per category")
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                       help="Number of images to generate per prompt")
    return parser.parse_args()

def generate_images(pipeline, prompt, num_images=1, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    output = pipeline(
        prompt,
        num_images_per_prompt=num_images,
        generator=generator,
        output_type="pil",
        guidance_scale=4,
    )
    return output.images[0]  # Return single image

def save_image_and_tensor(img, prefix, category, prompt_type, prompt_text, output_dir):
    """
    Save image as both PNG and PyTorch tensor with metadata.
    Args:
        img: PIL Image to save
        prefix: Model name (e.g., 'sd35')
        category: SugarCrepe category
        prompt_type: 'caption', 'caption2', or 'negative_caption'
        prompt_text: The actual prompt text used
        output_dir: Root output directory
    Returns:
        Tuple of (png_path, tensor_path)
    """
    # Create directories
    png_dir = os.path.join(output_dir, "png_images", prefix, category, prompt_type)
    tensor_dir = os.path.join(output_dir, "tensors", prefix, category, prompt_type)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)

    # Generate unique filename based on prompt hash
    prompt_hash = abs(hash(prompt_text)) % (10 ** 8)  # 8-digit hash
    filename = f"{prompt_type}_{prompt_hash}"

    # Save as PNG
    png_path = os.path.join(png_dir, f"{filename}.png")
    img.save(png_path)

    # Save as Tensor with metadata
    tensor_path = os.path.join(tensor_dir, f"{filename}.pt")
    img_tensor = ToTensor()(img)  # Convert PIL to tensor [C, H, W]
    
    # Create dictionary with image and metadata
    tensor_data = {
        'image': img_tensor,
        'model': prefix,
        'category': category,
        'prompt_type': prompt_type,
        'prompt_text': prompt_text,
        'png_path': png_path
    }
    
    torch.save(tensor_data, tensor_path)

    return png_path, tensor_path

def load_sugarcrepe_samples(samples_per_category):
    """Load samples from SugarCrepe++ dataset"""
    categories = [
        # "replace_attribute",
        # "replace_object",
        "replace_relation",
        # "swap_atribute",
        "swap_object"
    ]

    samples = []
    for category in categories:
        ds = load_dataset("Aman-J/SugarCrepe_pp", category)
        subset = ds['train'].select(range(samples_per_category))
        for example in subset:
            samples.append({
                'category': category,
                'caption': example['caption'],
                'caption2': example['caption2'],
                'negative_caption': example['negative_caption']
            })
    return pd.DataFrame(samples)

def run_experiment(samples_per_category=5, num_images_per_prompt=1):
    metadata = []
    df = load_sugarcrepe_samples(samples_per_category)

    # First process SD3.5
    print("Loading SD3.5...")
    sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.float16
    ).to(device)
    
    print("Generating images with SD3.5...")
    sd35_results = {}
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        category = row['category']
        for prompt_type in ['caption', 'caption2', 'negative_caption']:
            prompt_text = row[prompt_type]
            img = generate_images(sd35, prompt_text, num_images=num_images_per_prompt)
            # img = generate_images(sd35, prompt_text)
            
            png_path, tensor_path = save_image_and_tensor(
                img, 'sd35', category, prompt_type, prompt_text, output_dir
            )
            
            # Store results with composite key for matching with SD1.5 later
            key = (category, prompt_type, prompt_text)
            sd35_results[key] = {
                'png_path_sd35': png_path,
                'tensor_path_sd35': tensor_path,
                'image_index_sd35': idx  # Using dataframe index as image index
            }
    
    del sd35
    torch.cuda.empty_cache()

    # Then process SD1.5 with the same prompts
    print("Loading SD1.5...")
    sd15 = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to(device)
    
    print("Generating images with SD1.5...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        category = row['category']
        for prompt_type in ['caption', 'caption2', 'negative_caption']:
            prompt_text = row[prompt_type]
            img = generate_images(sd15, prompt_text, num_images=num_images_per_prompt)
            # img = generate_images(sd15, prompt_text)
            
            png_path, tensor_path = save_image_and_tensor(
                img, 'sd15', category, prompt_type, prompt_text, output_dir
            )
            
            # Get the corresponding SD3.5 results
            key = (category, prompt_type, prompt_text)
            sd35_data = sd35_results[key]
            
            # Combine all data for this prompt pair
            metadata.append({
                'category': category,
                'prompt_type': prompt_type,
                'prompt_text': prompt_text,
                'png_path_sd15': png_path,
                'png_path_sd35': sd35_data['png_path_sd35'],
                'tensor_path_sd15': tensor_path,
                'tensor_path_sd35': sd35_data['tensor_path_sd35'],
                'image_index_sd15': idx,
                'image_index_sd35': sd35_data['image_index_sd35']
            })
    
    del sd15
    torch.cuda.empty_cache()

    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"\nGenerated {len(metadata)*2} images total (SD1.5 + SD3.5)")
    print(f"Output saved to: {output_dir}")
    return metadata_df

# Run the experiment
if __name__ == "__main__":
    args = parse_args()
    metadata_df = run_experiment(
        samples_per_category=args.samples_per_category,
        num_images_per_prompt=args.num_images_per_prompt
    )

