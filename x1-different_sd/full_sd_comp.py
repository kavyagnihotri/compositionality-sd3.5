import os
import torch
import datetime
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from tqdm import tqdm
from huggingface_hub import InferenceClient, login
from torchvision.transforms import ToTensor
from diffusers import StableDiffusionPipeline, StableDiffusion3Pipeline
from PIL import Image
from sklearn.metrics.pairwise import cosine_similarity
from transformers import CLIPProcessor, CLIPModel
from datasets import load_dataset
from dotenv import load_dotenv

dotenv_path = os.path.join(os.path.dirname(__file__), '.env')
load_dotenv(dotenv_path=dotenv_path)  # Load from the specified path

#os.environ["HF_HUB_DISABLE_IMPLICIT_TOKEN"] = "1"  # Critical!
#os.environ["HF_HUB_DISABLE_GIT_CREDENTIALS"] = "1"

HUGGINGFACE_TOKEN = os.getenv("HF_TOKEN")
login(token=HUGGINGFACE_TOKEN, add_to_git_credential=False)

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

    # SD 3.5
    sd35 = StableDiffusion3Pipeline.from_pretrained(
        "stabilityai/stable-diffusion-3.5-large-turbo",
        torch_dtype=torch.float16 if torch.cuda.is_available() else torch.float32
    ).to(device)

    return sd15, sd35

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
        prefix: Model name (e.g., 'sd15')
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

# Save images to files
def save_images(images, prefix, category, prompt_type, index):
    os.makedirs(f"{output_dir}/{prefix}_images/{category}/{prompt_type}", exist_ok=True)
    filename = f"{output_dir}/{prefix}_images/{category}/{prompt_type}/{index:02d}.png"
    images.save(filename)

# Main experiment
def run_experiment():
    # Load models
    sd15, sd35 = load_models()

    # Load samples
    df = load_sugarcrepe_samples(samples_per_category=5) 
    print(f"Loaded {len(df)} samples from SugarCrepe++ dataset")

    # We'll generate 10 images for each caption type (caption, caption2, negative_caption)
    num_images_per_prompt = 10

    # Generate images with both models
    #print("Generating images...")
    #for model_name, pipeline in [('sd15', sd15), ('sd35', sd35)]:
    #    print(f"\nGenerating images with {model_name}...")
    #    for idx, row in tqdm(df.iterrows(), total=len(df)):
    #        category = row['category']
            
    #        # Generate from caption
    #        images = generate_images(pipeline, [row['caption']], num_images=num_images_per_prompt)
    #        for i, img in enumerate(images):
    #            save_images(img, model_name, category, "caption", i)
            
    #        # Generate from caption2
    #        images = generate_images(pipeline, [row['caption2']], num_images=num_images_per_prompt)
    #        for i, img in enumerate(images):
    #            save_images(img, model_name, category, "caption2", i)
            
    #        # Generate from negative_caption
    #        images = generate_images(pipeline, [row['negative_caption']], num_images=num_images_per_prompt)
    #        for i, img in enumerate(images):
    #            save_images(img, model_name, category, "negative_caption", i)

    #print("\nImage generation complete!")
    #print(f"Expected total images: {2 * 5 * 3 * 10} = 300 images (2 models x 5 categories x 3 prompt types x 10 images each)")

    ## Prepare data for similarity analysis
    #analysis_data = []
    #for model_name in ['sd15', 'sd35']:
    #    for category in df['category'].unique():
    #        for prompt_type in ['caption', 'caption2', 'negative_caption']:
    #            image_dir = f"{output_dir}/{model_name}_images/{category}/{prompt_type}"
    #            if os.path.exists(image_dir):
    #                for img_file in os.listdir(image_dir):
    #                    if img_file.endswith('.png'):
    #                        analysis_data.append({
    #                            'model': model_name,
    #                            'category': category,
    #                            'prompt_type': prompt_type,
    #                            'image_path': os.path.join(image_dir, img_file)
    #                        })
    
    #analysis_df = pd.DataFrame(analysis_data)
    #analysis_df.to_csv(f"{output_dir}/image_paths.csv", index=False)
    
    #return analysis_df

    for model_name, pipeline in [('sd15', sd15), ('sd35', sd35)]:
        print(f"\nGenerating images with {model_name}...")
        for idx, row in tqdm(df.iterrows(), total=len(df)):
            category = row['category']
            
            # Generate from caption
            images = generate_images(pipeline, [row['caption']], num_images=num_images_per_prompt)
            for i, img in enumerate(images):
                png_path, tensor_path = save_images_and_tensors(
                    img, model_name, category, "caption", i, output_dir
                )
            
            # Repeat for caption2 and negative_caption
            images = generate_images(pipeline, [row['caption2']], num_images=num_images_per_prompt)
            for i, img in enumerate(images):
                save_images_and_tensors(img, model_name, category, "caption2", i, output_dir)
                
            images = generate_images(pipeline, [row['negative_caption']], num_images=num_images_per_prompt)
            for i, img in enumerate(images):
                save_images_and_tensors(img, model_name, category, "negative_caption", i, output_dir)
    
    # Update metadata collection to include tensor paths
    analysis_data = []
    for model_name in ['sd15', 'sd35']:
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

# Function to load images for analysis
def load_images_for_analysis(analysis_df):
    images = []
    meta_data = []
    
    for _, row in analysis_df.iterrows():
        try:
            img = Image.open(row['image_path'])
            images.append(img)
            meta_data.append({
                'model': row['model'],
                'category': row['category'],
                'prompt_type': row['prompt_type'],
                'path': row['image_path']
            })
        except Exception as e:
            print(f"Error loading image {row['image_path']}: {e}")
    
    return images, pd.DataFrame(meta_data)

# Perform similarity analysis
def perform_similarity_analysis(analysis_df):
    print("\nPerforming similarity analysis...")
    
    # Load all images
    images, meta_df = load_images_for_analysis(analysis_df)
    
    if len(images) == 0:
        print("No images found for analysis")
        return None
    
    # Calculate similarity matrix
    similarity_matrix = calculate_structural_similarity(images)
    
    # Add similarity results to metadata
    results = []
    for i in range(len(meta_df)):
        for j in range(i+1, len(meta_df)):
            if meta_df.iloc[i]['category'] == meta_df.iloc[j]['category']:
                same_category = True
                same_prompt_type = (meta_df.iloc[i]['prompt_type'] == meta_df.iloc[j]['prompt_type'])
                same_model = (meta_df.iloc[i]['model'] == meta_df.iloc[j]['model'])
            else:
                same_category = False
                same_prompt_type = False
                same_model = False
            
            results.append({
                'image1': meta_df.iloc[i]['path'],
                'image2': meta_df.iloc[j]['path'],
                'model1': meta_df.iloc[i]['model'],
                'model2': meta_df.iloc[j]['model'],
                'category1': meta_df.iloc[i]['category'],
                'category2': meta_df.iloc[j]['category'],
                'prompt_type1': meta_df.iloc[i]['prompt_type'],
                'prompt_type2': meta_df.iloc[j]['prompt_type'],
                'same_category': same_category,
                'same_prompt_type': same_prompt_type,
                'same_model': same_model,
                'similarity': similarity_matrix[i, j]
            })
    
    results_df = pd.DataFrame(results)
    results_df.to_csv(f"{output_dir}/similarity_results.csv", index=False)
    
    return results_df

# Analyze results
def analyze_results(results_df):
    if results_df is None:
        return
    
    print("\nAnalyzing results...")
    
    with open(f"{output_dir}/analysis_results.txt", "w") as f:
        # Compare within same category vs different categories
        same_cat = results_df[results_df['same_category']]
        diff_cat = results_df[~results_df['same_category']]
        
        f.write(f"Average similarity within same category: {same_cat['similarity'].mean():.3f}\n")
        f.write(f"Average similarity between different categories: {diff_cat['similarity'].mean():.3f}\n")
        
        # Compare same prompt type vs different prompt types (within same category)
        same_cat_same_type = same_cat[same_cat['same_prompt_type']]
        same_cat_diff_type = same_cat[~same_cat['same_prompt_type']]
        
        f.write(f"\nWithin same category:\n")
        f.write(f"  Same prompt type similarity: {same_cat_same_type['similarity'].mean():.3f}\n")
        f.write(f"  Different prompt type similarity: {same_cat_diff_type['similarity'].mean():.3f}\n")
        
        # Compare models
        same_model = results_df[results_df['same_model']]
        diff_model = results_df[~results_df['same_model']]
        
        f.write(f"\nModel comparisons:\n")
        f.write(f"  Same model similarity: {same_model['similarity'].mean():.3f}\n")
        f.write(f"  Different model similarity: {diff_model['similarity'].mean():.3f}\n")
        
        # Compare models within same prompt type and category
        same_conditions = same_cat_same_type.copy()
        sd15_sd15 = same_conditions[(same_conditions['model1'] == 'sd15') & (same_conditions['model2'] == 'sd15')]
        sd35_sd35 = same_conditions[(same_conditions['model1'] == 'sd35') & (same_conditions['model2'] == 'sd35')]
        sd15_sd35 = same_conditions[(same_conditions['model1'] == 'sd15') & (same_conditions['model2'] == 'sd35')]
        
        f.write(f"\nSame category and prompt type comparisons:\n")
        f.write(f"  SD1.5 vs SD1.5 similarity: {sd15_sd15['similarity'].mean():.3f}\n")
        f.write(f"  SD3.5 vs SD3.5 similarity: {sd35_sd35['similarity'].mean():.3f}\n")
        f.write(f"  SD1.5 vs SD3.5 similarity: {sd15_sd35['similarity'].mean():.3f}\n")

# Run analysis
results_df = perform_similarity_analysis(analysis_df)
analyze_results(results_df)

print(f"\nAll outputs saved to directory: {output_dir}")
