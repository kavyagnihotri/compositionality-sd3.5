"""
Generate images from Standard, CLIC SD-3.5, and CogVLM SD-3.5.

Generates COCO-caption-conditioned images from each model for downstream
Clean-FID evaluation.
"""

import sys
sys.path.append("..")
sys.path.append("../z4-cogvlm_custom_sd35")

import argparse
import importlib.util
from pathlib import Path

import pandas as pd
import torch
from diffusers import StableDiffusion3Pipeline
from tqdm import tqdm


# Configuration
CAPTIONS_CSV = "./coco_5k_captions.csv"
OUTPUT_DIR = "./generated_images"

STANDARD_SD35_DIR = "../stable-diffusion-3.5-large-turbo"
CLIC_ENCODER_DIR = "../z0-custom_sd35_clic/converted_text_encoder"
COGVLM_ENCODER_DIR = "../z4-cogvlm_custom_sd35/converted_text_encoder_CogVLM"

MODEL_CONFIGS = {
    "standard": {
        "display_name": "Standard SD-3.5",
        "output_subdir": "standard_sd35_images",
    },
    "clic": {
        "display_name": "CLIC SD-3.5",
        "output_subdir": "clic_sd35_images",
    },
    "cogvlm": {
        "display_name": "CogVLM SD-3.5",
        "output_subdir": "cogvlm_sd35_images",
    },
}


def load_create_sd35_fn():
    """Load create_sd35_with_custom_encoder from the project custom module."""
    module_path = Path(__file__).resolve().parent.parent / "z4-cogvlm_custom_sd35" / "custom_sd35.py"
    spec = importlib.util.spec_from_file_location("custom_sd35_local", module_path)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    return module.create_sd35_with_custom_encoder


# Generation parameters
HEIGHT = 512
WIDTH = 512
NUM_STEPS = 4
GUIDANCE = 0.0
SEED = 42

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def parse_args():
    parser = argparse.ArgumentParser(
        description="Generate COCO images for selected SD-3.5 model(s)."
    )
    parser.add_argument(
        "--model",
        choices=["standard", "clic", "cogvlm", "all"],
        default="all",
        help="Which model to generate with.",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=SEED,
        help="Random seed for generation.",
    )
    parser.add_argument(
        "--guidance",
        type=float,
        default=GUIDANCE,
        help="CFG guidance scale. Turbo models should stay at 0.0.",
    )
    return parser.parse_args()


def generate_images(pipe, captions_df, model_name, output_dir, seed=42, guidance_scale=0.0):
    """
    Generate images from prompts and save PNG files.

    Args:
        pipe: Diffusion pipeline
        captions_df: DataFrame with 'caption' and 'image_id' columns
        model_name: Name for logging (e.g., "Standard SD-3.5")
        output_dir: Directory to store PNG outputs
        seed: Random seed for generation
        guidance_scale: CFG guidance scale for the pipeline

    Returns:
        dict with generated image metadata
    """
    print(f"Generating images with {model_name}")

    generator = torch.Generator(device=DEVICE).manual_seed(seed)

    out_dir = Path(output_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    num_saved = 0

    for idx, row in tqdm(captions_df.iterrows(), total=len(captions_df), desc=f"Generating ({model_name})"):
        caption = row["caption"]
        image_id = row["image_id"]

        with torch.no_grad():
            output = pipe(
                prompt=caption,
                num_inference_steps=NUM_STEPS,
                guidance_scale=guidance_scale,
                height=HEIGHT,
                width=WIDTH,
                generator=generator,
            )
            image = output.images[0]

        output_name = f"{idx:05d}_{int(image_id):012d}.png"
        image.save(out_dir / output_name)
        num_saved += 1

    print(f"  Saved {num_saved} PNGs to {out_dir}")

    return {
        "images_dir": str(out_dir),
        "num_images": num_saved,
        "height": HEIGHT,
        "width": WIDTH,
    }


def create_pipeline(model_key, create_sd35_with_custom_encoder):
    if model_key == "standard":
        return StableDiffusion3Pipeline.from_pretrained(
            STANDARD_SD35_DIR,
            torch_dtype=torch.float16,
            device_map=DEVICE,
        )
    if model_key == "clic":
        return create_sd35_with_custom_encoder(CLIC_ENCODER_DIR)
    if model_key == "cogvlm":
        return create_sd35_with_custom_encoder(COGVLM_ENCODER_DIR)
    raise ValueError(f"Unknown model key: {model_key}")


def main():
    args = parse_args()

    print("Image Generation:")
    print(f"Device: {DEVICE}")
    print(f"Parameters: {HEIGHT}x{WIDTH}, {NUM_STEPS} steps, guidance={args.guidance}")

    output_path = Path(OUTPUT_DIR)
    output_path.mkdir(parents=True, exist_ok=True)

    captions_df = pd.read_csv(CAPTIONS_CSV)
    print(f"Loaded {len(captions_df)} captions")

    create_sd35_with_custom_encoder = load_create_sd35_fn()

    prompts_path = output_path / "prompts.csv"
    captions_df.to_csv(prompts_path, index=False)
    print(f"Saved prompts to {prompts_path}")

    selected_models = list(MODEL_CONFIGS.keys()) if args.model == "all" else [args.model]
    print(f"Selected model(s): {', '.join(selected_models)}")

    results = {}
    for model_key in selected_models:
        cfg = MODEL_CONFIGS[model_key]
        print(f"LOADING {cfg['display_name'].upper()}")

        pipe = create_pipeline(model_key, create_sd35_with_custom_encoder)
        model_result = generate_images(
            pipe=pipe,
            captions_df=captions_df,
            model_name=cfg["display_name"],
            output_dir=output_path / cfg["output_subdir"],
            seed=args.seed,
            guidance_scale=args.guidance,
        )
        results[model_key] = model_result

        del pipe
        if torch.cuda.is_available():
            torch.cuda.empty_cache()

    print("GENERATION COMPLETE")
    for model_key in selected_models:
        cfg = MODEL_CONFIGS[model_key]
        model_result = results[model_key]
        print(f"{cfg['display_name']}: {model_result['num_images']} images")
        print(f"  Dir:  {model_result['images_dir']}")
    print("\nSelected PNG images generated and saved")


if __name__ == "__main__":
    main()
