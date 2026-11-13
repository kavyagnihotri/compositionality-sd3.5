#!/usr/bin/env python3
import os
import json
from safetensors.torch import load_file

def print_layers_and_weights(encoder_path):
    """Print all layers and their weights from text encoder."""
    print(f"TEXT ENCODER: {encoder_path}")
    
    if not os.path.exists(encoder_path):
        print(f"Path does not exist: {encoder_path}")
        return
    
    # Find safetensors file
    model_file = os.path.join(encoder_path, "model.safetensors")
    if not os.path.exists(model_file):
        print("model.safetensors not found")
        return
    
    print(f"Loading {model_file}...")
    weights = load_file(model_file)
    
    print(f"\nTotal layers: {len(weights)}\n")
    print(f"{'Layer Name':<70} {'Shape':<30} {'Parameters':<15}")
    
    total_params = 0
    for layer_name in sorted(weights.keys()):
        tensor = weights[layer_name]
        shape = str(tuple(tensor.shape))
        numel = tensor.numel()
        total_params += numel
        print(f"{layer_name:<70} {shape:<30} {numel:<15,}")
    
    print("-" * 115)
    print(f"{'TOTAL':<70} {'':<30} {total_params:<15,}")
    print(f"\nEstimated size: {total_params * 4 / (1024**3):.2f} GB (float32)")

if __name__ == "__main__":
    # Explore text_encoder_1 (named "text_encoder")
    encoder_path = "./stable-diffusion-3.5-large-turbo/text_encoder"
    print_layers_and_weights(encoder_path)