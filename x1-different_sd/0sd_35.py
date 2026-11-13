import os
import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import InferenceClient, login
from torchvision.transforms import ToTensor
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)  # Load from the specified path

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load model
def load_model():
    # SD 3.5
    sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3-large-turbo",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)
    return sd35

# CLIP model for structural similarity evaluation
clip_model = CLIPModel.from_pretrained("openai/clip-vit-base-patch32").to(device)
clip_processor = CLIPProcessor.from_pretrained("openai/clip-vit-base-patch32")

# Function to generate images
def generate_images(pipeline, prompts, num_images=1, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    for prompt in prompts:
        output = pipeline(
            prompt,
            num_images_per_prompt=num_images,
            generator=generator,
            output_type="pil"
        )
        images.extend(output.images)
    return images

# Load SugarCrepe++ dataset samples
def load_sugarcrepe_samples(samples_per_category=2):
    # Load the dataset
    categories = [
        "replace_attribute",
        "replace_object",
        "replace_relation",
        "swap_atribute",
        "swap_object"
    ]

    samples = []

    for category in categories:
        # Load the specific category
        ds = load_dataset("Aman-J/SugarCrepe_pp", category)

        # Take a subset of the data
        subset = ds['train'].select(range(samples_per_category))

        for example in subset:
            samples.append({
                'category': category,
                'caption': example['caption'],
                'caption2': example['caption2'],
                'negative_caption': example['negative_caption']
            })

    return pd.DataFrame(samples)

def save_images_and_tensors(img, prefix, category, prompt_type, index, output_dir):
    """
    Save images as both PNG and PyTorch tensors with metadata.
    Args:
        img: PIL Image to save
        prefix: Model name (e.g., 'sd35')
        category: SugarCrepe category (e.g., 'replace_attribute')
        prompt_type: 'caption', 'caption2', or 'negative_caption'
        index: Image index
        output_dir: Root output directory
    """
    # Create directories
    png_dir = os.path.join(output_dir, "png_images", prefix, category, prompt_type)
    tensor_dir = os.path.join(output_dir, "tensors", prefix, category, prompt_type)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)

    # --- Save as PNG ---
    png_path = os.path.join(png_dir, f"{index:03d}.png")
    img.save(png_path)

    # --- Save as Tensor ---
    tensor_path = os.path.join(tensor_dir, f"{index:03d}.pt")
    img_tensor = ToTensor()(img)  # Convert PIL to tensor [C, H, W]
    torch.save(img_tensor, tensor_path)

    return png_path, tensor_path

# Main experiment
def run_experiment():
    # Load model
    sd35 = load_model()

    # Load samples
    df = load_sugarcrepe_samples(samples_per_category=5)
    print(f"Loaded {len(df)} samples from SugarCrepe++ dataset")

    # We'll generate 10 images for each caption type (caption, caption2, negative_caption)
    num_images_per_prompt = 10

    model_name = 'sd35'
    print(f"\nGenerating images with {model_name}...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        category = row['category']

        # Generate from caption
        images = generate_images(sd35, [row['caption']], num_images=num_images_per_prompt)
        for i, img in enumerate(images):
            png_path, tensor_path = save_images_and_tensors(
                img, model_name, category, "caption", i, output_dir
            )

        # Generate from caption2
        images = generate_images(sd35, [row['caption2']], num_images=num_images_per_prompt)
        for i, img in enumerate(images):
            save_images_and_tensors(img, model_name, category, "caption2", i, output_dir)

        # Generate from negative_caption
        images = generate_images(sd35, [row['negative_caption']], num_images=num_images_per_prompt)
        for i, img in enumerate(images):
            save_images_and_tensors(img, model_name, category, "negative_caption", i, output_dir)

    # Update metadata collection to include tensor paths
    analysis_data = []
    for category in df['category'].unique():
        for prompt_type in ['caption', 'caption2', 'negative_caption']:
            png_dir = os.path.join(output_dir, "png_images", model_name, category, prompt_type)
            tensor_dir = os.path.join(output_dir, "tensors", model_name, category, prompt_type)

            if os.path.exists(png_dir):
                for img_file in os.listdir(png_dir):
                    if img_file.endswith('.png'):
                        index = int(img_file.split('.')[0])
                        analysis_data.append({
                            'model': model_name,
                            'category': category,
                            'prompt_type': prompt_type,
                            'png_path': os.path.join(png_dir, img_file),
                            'tensor_path': os.path.join(tensor_dir, f"{index:03d}.pt")
                        })

    analysis_df = pd.DataFrame(analysis_data)
    analysis_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    return analysis_df

# Run the experiment
analysis_df = run_experiment()

print(f"\nAll outputs saved to directory: {output_dir}")
