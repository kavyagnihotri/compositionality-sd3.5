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


torch.backends.cuda.enable_mem_efficient_sdp(True)  # Memory-efficient attention
torch.backends.cuda.enable_flash_sdp(True)         # Flash attention if available

# Additional memory management
os.environ["PYTORCH_CUDA_ALLOC_CONF"] = "expandable_segments:True"

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
    # SD 1.5 on GPU 0
    sd15 = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16
    ).to('cuda:0')

    # SD 3.5 on GPU 1
    sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-medium",
        #"stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.float16
    ).to('cuda:1')

    return sd15, sd35



def generate_images(pipeline, prompts, num_images=1, seed=42):
    generator = torch.Generator(device=device).manual_seed(seed)
    images = []
    # Process one prompt at a time with smaller batches
    for prompt in prompts:
        for i in range(0, num_images, 2):  # Batch size of 2
            output = pipeline(
                prompt,
                num_images_per_prompt=min(2, num_images-i),
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

def run_experiment(num_images_per_prompt=1):
    metadata = []
    df = load_sugarcrepe_samples(samples_per_category=5)

    # Load both models first
    print("Loading models...")
    sd15, sd35 = load_models()

    # Process SD3.5 on GPU 1
    print("Generating images with SD3.5...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        category = row['category']
        for prompt_type in ['caption', 'caption2', 'negative_caption']:
            prompt = row[prompt_type]
            images = generate_images(sd35, [prompt], num_images=num_images_per_prompt)

            for i, img in enumerate(images):
                png_path, tensor_path = save_images_and_tensors(
                    img, 'sd35', category, prompt_type, i, output_dir
                )
                metadata.append({
                    'model': 'sd35',
                    'category': category,
                    'prompt_type': prompt_type,
                    'prompt_text': prompt,
                    'png_path': png_path,
                    'tensor_path': tensor_path,
                    'image_index': i
                })

    # Clean up
    del sd35
    torch.cuda.empty_cache()

    print("Generating images with SD1.5...")
    for idx, row in tqdm(df.iterrows(), total=len(df)):
        category = row['category']
        for prompt_type in ['caption', 'caption2', 'negative_caption']:
            prompt = row[prompt_type]
            images = generate_images(sd15, [prompt], num_images=num_images_per_prompt)
            
            for i, img in enumerate(images):
                png_path, tensor_path = save_images_and_tensors(
                    img, 'sd15', category, prompt_type, i, output_dir
                )
                metadata.append({
                    'model': 'sd15',
                    'category': category,
                    'prompt_type': prompt_type,
                    'prompt_text': prompt,
                    'png_path': png_path,
                    'tensor_path': tensor_path,
                    'image_index': i
                })
    del sd15
    torch.cuda.empty_cache()


    # Save metadata
    metadata_df = pd.DataFrame(metadata)
    metadata_df.to_csv(os.path.join(output_dir, "metadata.csv"), index=False)
    print(f"\nGenerated {len(metadata)} images total")
    print(f"Output saved to: {output_dir}")
    return metadata_df


# Run the experiment
metadata_df = run_experiment()
