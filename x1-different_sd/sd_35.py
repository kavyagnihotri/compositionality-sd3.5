import os
import torch
import datetime
import pandas as pd
from tqdm import tqdm
from huggingface_hub import login
from torchvision.transforms import ToTensor
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from datasets import load_dataset
from dotenv import load_dotenv

# Load environment variables
dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)
HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"sd35_output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Initialize device and model
device = "cuda" if torch.cuda.is_available() else "cpu"

def load_model():
    pipe = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.bfloat16 if torch.cuda.is_available() else torch.float32
    )
    return pipe.to(device)

def load_prompts(samples_per_category=2):
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

def save_output(img, prompt, category, prompt_type, index, output_dir):
    # Create directories
    img_dir = os.path.join(output_dir, "images", category, prompt_type)
    tensor_dir = os.path.join(output_dir, "tensors", category, prompt_type)
    os.makedirs(img_dir, exist_ok=True)
    os.makedirs(tensor_dir, exist_ok=True)
    
    # Save image
    img_path = os.path.join(img_dir, f"{index:03d}.jpg")
    img.save(img_path, quality=95)
    
    # Save tensor
    tensor_path = os.path.join(tensor_dir, f"{index:03d}.pt")
    torch.save(ToTensor()(img), tensor_path)
    
    return {
        'category': category,
        'prompt_type': prompt_type,
        'prompt': prompt,
        'image_path': img_path,
        'tensor_path': tensor_path,
        'index': index
    }

def generate_images():
    model = load_model()
    prompts_df = load_prompts(samples_per_category=5)
    num_images_per_prompt = 10
    metadata = []
    
    print("Generating images...")
    for _, row in tqdm(prompts_df.iterrows(), total=len(prompts_df)):
        category = row['category']
        
        for prompt_type in ['caption', 'caption2', 'negative_caption']:
            prompt = row[prompt_type]
            images = model(
                prompt,
                num_images_per_prompt=num_images_per_prompt,
                num_inference_steps=4,
                guidance_scale=0.0,
                output_type="pil"
            ).images
            
            for i, img in enumerate(images):
                metadata.append(
                    save_output(img, prompt, category, prompt_type, i, output_dir)
                )
    
    # Save metadata
    pd.DataFrame(metadata).to_csv(os.path.join(output_dir, "prompts_metadata.csv"), index=False)
    print(f"\nGeneration complete! Output saved to: {output_dir}")

if __name__ == "__main__":
    generate_images()
