import os
import torch
import datetime
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

def load_models():
    """Load Stable Diffusion models"""
    # SD 1.5
    sd15 = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    # SD 3.5
    sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    return sd15, sd35

def generate_images(pipeline, prompts, num_images=1, seed=42):
    """Generate images from prompts"""
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

def save_images_and_tensors(img, prefix, category, prompt_type, index, output_dir):
    """
    Save images as both PNG and PyTorch tensors with metadata.
    Args:
        img: PIL Image to save
        prefix: Model name (e.g., 'sd15')
        category: SugarCrepe category
        prompt_type: 'caption', 'caption2', or 'negative_caption'
        index: Image index
        output_dir: Root output directory
    """
    # Create directories
    png_dir = os.path.join(output_dir, "png_images", prefix, category, prompt_type)
    tensor_dir = os.path.join(output_dir, "tensors", prefix, category, prompt_type)
    os.makedirs(png_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)

    # Save as PNG
    png_path = os.path.join(png_dir, f"{index:03d}.png")
    img.save(png_path)

    # Save as Tensor
    tensor_path = os.path.join(tensor_dir, f"{index:03d}.pt")
    img_tensor = ToTensor()(img)  # Convert PIL to tensor [C, H, W]
    torch.save(img_tensor, tensor_path)

    return png_path, tensor_path

def load_sugarcrepe_samples(samples_per_category=5):
    """Load samples from SugarCrepe++ dataset"""
    categories = [
        "replace_attribute",
        "replace_object",
        "replace_relation",
        "swap_atribute",
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

def run_experiment():
    """Main function to generate and save images"""
    # Load models and samples
    sd15, sd35 = load_models()
    df = load_sugarcrepe_samples(samples_per_category=5)
    print(f"Loaded {len(df)} samples from SugarCrepe++ dataset")
    
    num_images_per_prompt = 10
    metadata = []

    for model_name, pipeline in [('sd15', sd15), ('sd35', sd35)]:
        print(f"\nGenerating images with {model_name}...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            category = row['category']

            # Process each prompt type
            for prompt_type in ['caption', 'caption2', 'negative_caption']:
                prompt = row[prompt_type]
                images = generate_images(pipeline, [prompt], num_images=num_images_per_prompt)
                
                for i, img in enumerate(images):
                    png_path, tensor_path = save_images_and_tensors(
                        img, model_name, category, prompt_type, i, output_dir
                    )
                    
                    # Record metadata
                    metadata.append({
                        'model': model_name,
                        'category': category,
                        'prompt_type': prompt_type,
                        'prompt_text': prompt,
                        'png_path': png_path,
                        'tensor_path': tensor_path,
                        'image_index': i
                    })

    # Save metadata to CSV
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"\nGenerated {len(metadata)} images total")
    print(f"Output saved to: {output_dir}")
    return metadata_df

# Run the experiment
metadata_df = run_experiment()
