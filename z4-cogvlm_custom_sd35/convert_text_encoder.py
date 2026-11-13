"""
Convert CLIC CogVLM OpenCLIP Text Encoder to SD3.5 HuggingFace Format
  1. Split fused QKV attention matrices into separate Q, K, V projections
  2. Rename MLP layers (c_fc > fc1, c_proj > fc2)
  3. Transpose text_projection matrix
  4. Configure QuickGELU activation (critical for CLIC)
"""

import torch
from pathlib import Path
from transformers import CLIPTextModelWithProjection, CLIPTextConfig, CLIPTokenizer
import open_clip


OPENCLIP_CHECKPOINT = 'hf-hub:nmndeep/CLIC-ViT-L-14-224-CogVLM'
CLIC_LOCAL_DIR = "../CLIC-ViT-L-14-224-CogVLM"
OUTPUT_DIR = "./converted_text_encoder_CogVLM"
NUM_LAYERS = 12  # SD3.5 text_encoder uses 12 layers


def convert_clic_to_sd35_format(output_dir=OUTPUT_DIR):
    """
    Convert CogVLM text encoder to SD3.5 format.
    Returns: tuple: (converted_model, weight_mapping_dict)
    """
    print("Loading CLIC CogVLM OpenCLIP model...")
    openclip_model, _, _ = open_clip.create_model_and_transforms(OPENCLIP_CHECKPOINT)
    state_dict = openclip_model.state_dict()
    
    # Detect architecture from weights
    text_width = state_dict['token_embedding.weight'].shape[1]
    vocab_size = state_dict['token_embedding.weight'].shape[0]
    context_length = state_dict['positional_embedding'].shape[0]
    
    print(f"Architecture: {text_width}D hidden, {vocab_size} vocab, {context_length} context")
    
    # Create HuggingFace config
    config = CLIPTextConfig(
        hidden_size=text_width,
        intermediate_size=text_width * 4,
        num_hidden_layers=NUM_LAYERS,
        num_attention_heads=text_width // 64,
        max_position_embeddings=context_length,
        vocab_size=vocab_size,
        projection_dim=text_width,
        layer_norm_eps=1e-5,
        hidden_act="quick_gelu",
        bos_token_id=0,
        eos_token_id=2,
    )
    
    # Initialize model and prepare for weight mapping
    hf_model = CLIPTextModelWithProjection(config)
    new_state_dict = {}
    weight_mapping = []
    
    print("Converting weights...")
    
    # Process each weight with surgery
    for old_key, value in state_dict.items():
        # Skip vision encoder and logit_scale
        if old_key.startswith('visual.') or old_key == 'logit_scale':
            continue
        
        # EMBEDDINGS
        if old_key == 'token_embedding.weight':
            new_key = 'text_model.embeddings.token_embedding.weight'
            new_state_dict[new_key] = value
            weight_mapping.append((old_key, new_key))
            
        elif old_key == 'positional_embedding':
            new_key = 'text_model.embeddings.position_embedding.weight'
            new_state_dict[new_key] = value
            weight_mapping.append((old_key, new_key))
        
        # TRANSFORMER LAYERS
        elif old_key.startswith('transformer.resblocks.'):
            layer_idx = int(old_key.split('.')[2])
            
            # Only use first NUM_LAYERS for SD3.5
            if layer_idx >= NUM_LAYERS:
                continue
            
            # ATTENTION: Split fused QKV weights
            if '.attn.in_proj_weight' in old_key:
                chunk_size = value.shape[0] // 3
                q_key = f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight'
                k_key = f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight'
                v_key = f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight'
                
                new_state_dict[q_key] = value[:chunk_size, :]
                new_state_dict[k_key] = value[chunk_size:2*chunk_size, :]
                new_state_dict[v_key] = value[2*chunk_size:, :]
                
                weight_mapping.append((old_key, f"{q_key} [Q slice]"))
                weight_mapping.append((old_key, f"{k_key} [K slice]"))
                weight_mapping.append((old_key, f"{v_key} [V slice]"))
                
            elif '.attn.in_proj_bias' in old_key:
                chunk_size = value.shape[0] // 3
                q_key = f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias'
                k_key = f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias'
                v_key = f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias'
                
                new_state_dict[q_key] = value[:chunk_size]
                new_state_dict[k_key] = value[chunk_size:2*chunk_size]
                new_state_dict[v_key] = value[2*chunk_size:]
                
                weight_mapping.append((old_key, f"{q_key} [Q slice]"))
                weight_mapping.append((old_key, f"{k_key} [K slice]"))
                weight_mapping.append((old_key, f"{v_key} [V slice]"))
                
            elif '.attn.out_proj.weight' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
                
            elif '.attn.out_proj.bias' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
            
            # LAYER NORMS
            elif '.ln_1.weight' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.layer_norm1.weight'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
                
            elif '.ln_1.bias' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.layer_norm1.bias'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
                
            elif '.ln_2.weight' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.layer_norm2.weight'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
                
            elif '.ln_2.bias' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.layer_norm2.bias'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
            
            # MLP: Rename c_fc to fc1, c_proj to fc2
            elif '.mlp.c_fc.weight' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.mlp.fc1.weight'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
                
            elif '.mlp.c_fc.bias' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.mlp.fc1.bias'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
                
            elif '.mlp.c_proj.weight' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.mlp.fc2.weight'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
                
            elif '.mlp.c_proj.bias' in old_key:
                new_key = f'text_model.encoder.layers.{layer_idx}.mlp.fc2.bias'
                new_state_dict[new_key] = value
                weight_mapping.append((old_key, new_key))
        
        # FINAL LAYERS
        elif old_key == 'ln_final.weight':
            new_key = 'text_model.final_layer_norm.weight'
            new_state_dict[new_key] = value
            weight_mapping.append((old_key, new_key))
            
        elif old_key == 'ln_final.bias':
            new_key = 'text_model.final_layer_norm.bias'
            new_state_dict[new_key] = value
            weight_mapping.append((old_key, new_key))
        
        # TEXT PROJECTION: Transpose the matrix
        elif old_key == 'text_projection':
            new_key = 'text_projection.weight'
            new_state_dict[new_key] = value.T  # TRANSPOSE
            weight_mapping.append((old_key, f"{new_key} [transposed]"))
    
    # Load converted weights
    missing_keys, unexpected_keys = hf_model.load_state_dict(new_state_dict, strict=False)
    
    if missing_keys:
        print(f"\n WARNING: {len(missing_keys)} weights not mapped from CLIC (remain randomly initialized):")
        for key in missing_keys:
            print(f"  {key}")
    
    if unexpected_keys:
        print(f"\n WARNING: {len(unexpected_keys)} unexpected keys in state_dict:")
        for key in unexpected_keys:
            print(f"  {key}")
    
    # Save model
    output_path = Path(output_dir)
    output_path.mkdir(parents=True, exist_ok=True)
    hf_model.save_pretrained(output_path)
    config.save_pretrained(output_path)
    
    # Copy tokenizer from local CLIC directory
    clic_dir = Path(CLIC_LOCAL_DIR)
    if clic_dir.exists():
        tokenizer = CLIPTokenizer.from_pretrained(str(clic_dir))
    else:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    tokenizer.model_max_length = 77
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    tokenizer.save_pretrained(output_path)
    
    print(f"\nSaved converted model to: {output_path}")
    
    return hf_model, weight_mapping


def print_weight_mapping(weight_mapping):
    """Print a formatted table of weight name mappings"""
    print("\nWEIGHT MAPPING: OpenCLIP to HuggingFace SD3.5 Format")
    print(f"{'OLD NAME (OpenCLIP)':<60} {'NEW NAME (HuggingFace)':<60}")
    
    for old_name, new_name in weight_mapping:
        print(f"{old_name:<60} {new_name:<60}")
    
    print(f"\nTotal weight transformations: {len(weight_mapping)}")


if __name__ == "__main__":
    model, mapping = convert_clic_to_sd35_format()
    print_weight_mapping(mapping)
