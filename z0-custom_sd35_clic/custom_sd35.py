"""
Replace Text Encoder 1 of Stable Diffusion 3.5 Large Turbo with Custom OpenCLIP (CLIC)

Uses pre-converted CLIC text encoder for SD3.5 pipeline integration.

Architecture:
  SD3.5 has 3 text encoders:
    - text_encoder (CLIP-L, 768-dim) <- REPLACED with custom CLIC
    - text_encoder_2 (CLIP-G, 1280-dim) <- Original
    - text_encoder_3 (T5-XXL, 4096-dim) <- Original
"""

import torch
import torch.nn as nn
from pathlib import Path
from transformers import (
    CLIPTextModelWithProjection, 
    CLIPTokenizer,
    T5EncoderModel,
    T5TokenizerFast
)
from diffusers import (
    StableDiffusion3Pipeline,
    SD3Transformer2DModel,
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler
)


# ==================== CONFIGURATION ====================
LOCAL_SD_DIR = "/mnt/lustre/work/hein/hof217/stable_diffusion/stable-diffusion-3.5-large-turbo"
CONVERTED_MODEL_DIR = "./converted_text_encoder"

DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


# ==================== QUICKGELU ACTIVATION ====================

class QuickGELU(nn.Module):
    """
    QuickGELU activation function used in OpenCLIP.
    Formula: x * sigmoid(1.702 * x)
    
    This is DIFFERENT from standard GELU and must be used for OpenCLIP models.
    """
    def forward(self, x: torch.Tensor):
        return x * torch.sigmoid(1.702 * x)


def patch_quickgelu_activation(model):
    """
    Manually replace all activation functions in the model with QuickGELU.
    
    HuggingFace's CLIPTextModel doesn't properly support quick_gelu via config,
    so we need to manually patch the MLP activation layers.
    
    Args:
        model: CLIPTextModelWithProjection instance
    """
    print("\nPatching QuickGELU activation functions...")
    
    # Replace activation in all encoder layers
    for i, layer in enumerate(model.text_model.encoder.layers):
        # The MLP has an activation_fn attribute
        if hasattr(layer.mlp, 'activation_fn'):
            layer.mlp.activation_fn = QuickGELU()
    
    print(f"  Patched {len(model.text_model.encoder.layers)} layers with QuickGELU")
    return model


# ==================== PIPELINE INTEGRATION ====================

def create_sd35_with_custom_encoder(converted_encoder_dir=None):
    """
    Create SD3.5 pipeline with component-based loading.
    
    Loads components individually to explicitly control encoder replacement.
    
    SD3.5 Architecture:
      - text_encoder (CLIP-L) ← REPLACED with custom CLIC
      - text_encoder_2 (CLIP-G) ← Keep original
      - text_encoder_3 (T5-XXL) ← Keep original
    
    Args:
        converted_encoder_dir: Path to converted model (default: CONVERTED_MODEL_DIR)
        
    Returns:
        StableDiffusion3Pipeline with custom text_encoder
    """
    print("\nLOADING SD3.5 PIPELINE WITH CUSTOM TEXT ENCODER")
    
    if converted_encoder_dir is None:
        converted_encoder_dir = CONVERTED_MODEL_DIR
    
    print("\nLoading SD3.5 Components Individually...")
    
    # Load core diffusion components (unchanged)
    print("  [1/9] Loading Transformer (MMDiT)...")
    transformer = SD3Transformer2DModel.from_pretrained(
        LOCAL_SD_DIR, 
        subfolder="transformer",
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    print("  [2/9] Loading VAE...")
    vae = AutoencoderKL.from_pretrained(
        LOCAL_SD_DIR,
        subfolder="vae",
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    print("  [3/9] Loading Scheduler...")
    scheduler = FlowMatchEulerDiscreteScheduler.from_pretrained(
        LOCAL_SD_DIR,
        subfolder="scheduler"
    )
    
    # Load text encoders (1 custom, 2 original)
    print("  [4/9] Loading TEXT ENCODER 1 (CUSTOM CLIP-L)...")
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        converted_encoder_dir,
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    # CRITICAL: Patch QuickGELU after loading from disk
    text_encoder = patch_quickgelu_activation(text_encoder)
    
    print("  [5/9] Loading TEXT ENCODER 2 (CLIP-G - Original)...")
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        LOCAL_SD_DIR,
        subfolder="text_encoder_2",
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    print("  [6/9] Loading TEXT ENCODER 3 (T5-XXL - Original)...")
    text_encoder_3 = T5EncoderModel.from_pretrained(
        LOCAL_SD_DIR,
        subfolder="text_encoder_3",
        torch_dtype=torch.float16,
        device_map=DEVICE
    )
    
    # Load tokenizers (1 custom, 2 original)
    print("  [7/9] Loading TOKENIZER 1 (Custom)...")
    tokenizer = CLIPTokenizer.from_pretrained(converted_encoder_dir)
    
    print("  [8/9] Loading TOKENIZER 2 (Original)...")
    tokenizer_2 = CLIPTokenizer.from_pretrained(
        LOCAL_SD_DIR,
        subfolder="tokenizer_2"
    )
    
    print("  [9/9] Loading TOKENIZER 3 (Original)...")
    tokenizer_3 = T5TokenizerFast.from_pretrained(
        LOCAL_SD_DIR,
        subfolder="tokenizer_3"
    )
    
    # Assemble pipeline (all components already on GPU)
    print("\nAssembling Pipeline...")
    pipe = StableDiffusion3Pipeline(
        transformer=transformer,
        vae=vae,
        scheduler=scheduler,
        text_encoder=text_encoder,           # ← CUSTOM (on GPU)
        text_encoder_2=text_encoder_2,       # ← ORIGINAL (on GPU)
        text_encoder_3=text_encoder_3,       # ← ORIGINAL (on GPU)
        tokenizer=tokenizer,                 # ← CUSTOM (CPU)
        tokenizer_2=tokenizer_2,             # ← ORIGINAL (CPU)
        tokenizer_3=tokenizer_3              # ← ORIGINAL (CPU)
    )
    
    print(f"Pipeline assembled with custom text_encoder (components on {DEVICE})")
    
    return pipe


# ==================== TESTING UTILITIES ====================

def test_generation(pipe, prompts=None, output_dir="./test_outputs"):
    """
    Test the pipeline with sample prompts.
    
    Args:
        pipe: StableDiffusion3Pipeline
        prompts: List of test prompts
        output_dir: Directory to save outputs
    """
    print("\nTESTING IMAGE GENERATION")
    
    if prompts is None:
        prompts = [
            "a red ball above a blue cube",
            "a cat sitting to the left of a dog",
            "a tree below the clouds",
            "a coffee cup on top of a wooden table"
        ]
    
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    
    for i, prompt in enumerate(prompts):
        print(f"\n[{i+1}/{len(prompts)}] Generating: '{prompt}'")
        
        with torch.no_grad():
            image = pipe(
                prompt=prompt,
                num_inference_steps=28,
                guidance_scale=3.5,
                height=512,
                width=512,
            ).images[0]
        
        save_path = output_path / f"test_{i+1:02d}.png"
        image.save(save_path)
        print(f"  Saved to: {save_path}")
    
    print("\nAll test images generated")