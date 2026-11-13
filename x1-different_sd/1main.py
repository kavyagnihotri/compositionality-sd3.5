import torch
from diffusers import StableDiffusionPipeline, StableDiffusionXLPipeline
from PIL import Image
import numpy as np
import pandas as pd
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
import os
from tqdm import tqdm
import matplotlib.pyplot as plt
from datasets import load_dataset
import datetime

# Create output directory
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
output_dir = f"output_{timestamp}"
os.makedirs(output_dir, exist_ok=True)

# Initialize device
device = "cuda" if torch.cuda.is_available() else "cpu"

# Load models
def load_models():
    # SD 1.5
    sd15 = StableDiffusionPipeline.from_pretrained(
        "runwayml/stable-diffusion-v1-5",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    # SD XL (as proxy for future 3.5)
    sdxl = StableDiffusionXLPipeline.from_pretrained(
        "stabilityai/stable-diffusion-xl-base-1.0",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    return sd15, sdxl

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
        images.append(output.images[0])
    return images

# Function to calculate structural similarity using CLIP embeddings
def calculate_structural_similarity(images):
    # Preprocess images
    inputs = clip_processor(images=images, return_tensors="pt", padding=True).to(device)

    # Get image embeddings
    with torch.no_grad():
        image_features = clip_model.get_image_features(**inputs)

    # Calculate cosine similarity matrix
    similarity_matrix = cosine_similarity(image_features.cpu().numpy())
    return similarity_matrix

# Load SugarCrepe++ dataset samples
def load_sugarcrepe_samples(samples_per_category=10):
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
                'negative_caption': example['negative_caption'],
                # We'll use caption as the main prompt for generation
                'prompt': example['caption']
            })

    return pd.DataFrame(samples)

# Save images to files
def save_images(images, prefix, df):
    os.makedirs(f"{output_dir}/{prefix}_images", exist_ok=True)
    for i, (image, prompt) in enumerate(zip(images, df['prompt'])):
        # Create a safe filename
        safe_prompt = "".join([c for c in prompt[:50] if c.isalnum() or c in (' ', '-', '_')]).rstrip()
        filename = f"{output_dir}/{prefix}_images/{i:02d}_{safe_prompt}.png"
        image.save(filename)

# Main experiment
def run_experiment():
    # Load models
    sd15, sdxl = load_models()

    # Load samples
    df = load_sugarcrepe_samples(samples_per_category=4)  # 4 samples per category * 5 categories = 20 total
    prompts = df['prompt'].tolist()

    # Generate images with both models
    print("Generating images with SD 1.5...")
    sd15_images = generate_images(sd15, prompts)

    print("Generating images with SD XL...")
    sdxl_images = generate_images(sdxl, prompts)

    # Save images
    save_images(sd15_images, "sd15", df)
    save_images(sdxl_images, "sdxl", df)

    # Calculate structural similarities
    print("Calculating structural similarities...")
    sd15_sim = calculate_structural_similarity(sd15_images)
    sdxl_sim = calculate_structural_similarity(sdxl_images)

    # Analyze results
    results = []
    for i in range(len(prompts)):
        for j in range(i+1, len(prompts)):
            if df.iloc[i]['category'] == df.iloc[j]['category']:
                same_category = True
                # For SugarCrepe++, we can compare if they share the same modification type
                same_subtype = (df.iloc[i]['caption'] == df.iloc[j]['caption'])  # Simplified comparison
            else:
                same_category = False
                same_subtype = False

            results.append({
                'pair': (i, j),
                'same_category': same_category,
                'same_subtype': same_subtype,
                'sd15_similarity': sd15_sim[i, j],
                'sdxl_similarity': sdxl_sim[i, j]
            })

    results_df = pd.DataFrame(results)

    # Save results to CSV
    df.to_csv(f"{output_dir}/prompts.csv", index=False)
    results_df.to_csv(f"{output_dir}/similarity_results.csv", index=False)

    return df, results_df, sd15_images, sdxl_images

# Run the experiment
df, results_df, sd15_images, sdxl_images = run_experiment()

# Analysis functions
def analyze_category_differences(results_df):
    # Compare similarity scores within vs between categories
    within_cat = results_df[results_df['same_category']]['sd15_similarity'].mean()
    between_cat = results_df[~results_df['same_category']]['sd15_similarity'].mean()

    within_cat_xl = results_df[results_df['same_category']]['sdxl_similarity'].mean()
    between_cat_xl = results_df[~results_df['same_category']]['sdxl_similarity'].mean()

    # Save analysis to a text file
    with open(f"{output_dir}/analysis_results.txt", "w") as f:
        f.write(f"SD1.5 Within-category similarity: {within_cat:.3f}\n")
        f.write(f"SD1.5 Between-category similarity: {between_cat:.3f}\n")
        f.write(f"SDXL Within-category similarity: {within_cat_xl:.3f}\n")
        f.write(f"SDXL Between-category similarity: {between_cat_xl:.3f}\n")

def analyze_model_performance(results_df):
    # Compare similarity scores between models
    same_subtype = results_df[results_df['same_subtype']]
    diff_subtype = results_df[results_df['same_category'] & ~results_df['same_subtype']]

    # Append to analysis file
    with open(f"{output_dir}/analysis_results.txt", "a") as f:
        f.write("\nSame subtype pairs:\n")
        f.write(f"SD1.5 avg similarity: {same_subtype['sd15_similarity'].mean():.3f}\n")
        f.write(f"SDXL avg similarity: {same_subtype['sdxl_similarity'].mean():.3f}\n")

        f.write("\nDifferent subtype (same category) pairs:\n")
        f.write(f"SD1.5 avg similarity: {diff_subtype['sd15_similarity'].mean():.3f}\n")
        f.write(f"SDXL avg similarity: {diff_subtype['sdxl_similarity'].mean():.3f}\n")

# Save visualizations as images
def save_examples(df, sd15_images, sdxl_images, num_examples=3):
    fig, axes = plt.subplots(num_examples, 3, figsize=(15, 5*num_examples))

    for i in range(num_examples):
        # Show prompt
        prompt_text = f"Category: {df.iloc[i]['category']}\nPrompt: {df.iloc[i]['prompt']}"
        axes[i, 0].text(0.5, 0.5, prompt_text, ha='center', va='center', wrap=True)
        axes[i, 0].axis('off')
        axes[i, 0].set_title("Prompt")

        # Show SD1.5 image
        axes[i, 1].imshow(sd15_images[i])
        axes[i, 1].axis('off')
        axes[i, 1].set_title("SD1.5")

        # Show SDXL image
        axes[i, 2].imshow(sdxl_images[i])
        axes[i, 2].axis('off')
        axes[i, 2].set_title("SDXL")

    plt.tight_layout()
    plt.savefig(f"{output_dir}/example_comparisons.png")
    plt.close()

# Run analyses
analyze_category_differences(results_df)
analyze_model_performance(results_df)
save_examples(df, sd15_images, sdxl_images)

print(f"All outputs saved to directory: {output_dir}")
