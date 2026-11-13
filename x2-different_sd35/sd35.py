import os
import torch
import datetime
import argparse
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from torchvision.transforms import ToTensor
from diffusers import StableDiffusion3Pipeline
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Login to Hugging Face
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"sd35_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--num_images_per_prompt", type=int, default=1,
                       help="Number of images to generate per prompt")
    parser.add_argument("--prompt_file", type=str, default="prompts.txt",
                       help="Path to prompt file (one prompt per line)")
    parser.add_argument("--output_csv", type=str, default="metadata.csv",
                       help="CSV file to store metadata (appends if exists)")
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


def save_image_and_tensor(img, prefix, prompt_text, output_dir):
    """
    Save image as both PNG and PyTorch tensor with metadata.
    Args:
        img: PIL Image to save
        prefix: Model name (e.g., 'sd35')
        prompt_text: The actual prompt text used
        output_dir: Root output directory
    Returns:
        Tuple of (png_path, tensor_path)
    """
    # Create directories
    png_dir = os.path.join(output_dir, "png_images", prefix)
    tensor_dir = os.path.join(output_dir, "tensors", prefix)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)

    # Generate unique filename based on prompt hash
    prompt_hash = abs(hash(prompt_text)) % (10 ** 8)  # 8-digit hash
    filename = f"prompt_{prompt_hash}"

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
        'prompt_text': prompt_text,
        'png_path': png_path
    }
    
    torch.save(tensor_data, tensor_path)

    return png_path, tensor_path


def load_prompts_from_file(prompt_file):
    """Load prompts from a file, one per line."""
    with open(prompt_file, 'r') as f:
        prompts = [line.strip() for line in f if line.strip()]
    return prompts


def get_last_image_index(csv_path):
    """Get the last image_index_sd35 from the CSV file, or -1 if not found."""
    if not os.path.exists(csv_path):
        return -1
    try:
        df = pd.read_csv(csv_path)
        if 'image_index_sd35' in df.columns and not df.empty:
            return int(df['image_index_sd35'].max())
        else:
            return -1
    except Exception:
        return -1


def run_experiment(num_images_per_prompt=1, prompt_file="prompt.txt", output_csv="metadata.csv"):
    prompts = load_prompts_from_file(prompt_file)
    print(f"Loaded {len(prompts)} prompts from {prompt_file}")

    print("Loading SD3.5...")
    sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.float16
    ).to(device)

    print("Generating images with SD3.5...")
    last_idx = get_last_image_index(os.path.join(output_dir, output_csv))
    image_index_sd35 = last_idx + 1
    rows = []
    for prompt_text in tqdm(prompts):
        for _ in range(num_images_per_prompt):
            img = generate_images(sd35, prompt_text, num_images=1)
            png_path, tensor_path = save_image_and_tensor(
                img, 'sd35', prompt_text, output_dir
            )
            rows.append({
                'prompt_text': prompt_text,
                'png_path_sd35': png_path,
                'tensor_path_sd35': tensor_path,
                'image_index_sd35': image_index_sd35
            })
            image_index_sd35 += 1

    del sd35
    torch.cuda.empty_cache()

    # Append to CSV
    csv_path = os.path.join(output_dir, output_csv)
    df_new = pd.DataFrame(rows)
    if os.path.exists(csv_path):
        df_new.to_csv(csv_path, mode='a', header=False, index=False)
    else:
        df_new.to_csv(csv_path, mode='w', header=True, index=False)
    print(f"Output saved to: {csv_path}")
    return df_new


# Run the experiment
if __name__ == "__main__":
    args = parse_args()
    run_experiment(
        num_images_per_prompt=args.num_images_per_prompt,
        prompt_file=args.prompt_file,
        output_csv=args.output_csv
    )