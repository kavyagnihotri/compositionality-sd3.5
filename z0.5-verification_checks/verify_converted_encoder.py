"""
Verify Converted Text Encoder against SD3.5 Original
1. Check if converted encoder has all required layers matching SD3.5 structure
2. Computes Frobenius norm (L2 distance) between corresponding weights
3. Generates visualization of weight differences per layer
"""

import torch
from pathlib import Path
from transformers import CLIPTextModelWithProjection
import matplotlib.pyplot as plt
import numpy as np
import csv


# Configuration
ORIGINAL_SD35_DIR = "/mnt/lustre/work/hein/hof217/stable_diffusion/stable-diffusion-3.5-large-turbo"
CONVERTED_ENCODER_DIR = "./converted_text_encoder"
OUTPUT_PLOT = "./weight_comparison.png"
OUTPUT_CSV = "./weight_comparison.csv"


def load_models():
    """Load both original and converted text encoders"""

    original = CLIPTextModelWithProjection.from_pretrained(
        ORIGINAL_SD35_DIR,
        subfolder="text_encoder",
        dtype=torch.float32
    )
    
    converted = CLIPTextModelWithProjection.from_pretrained(
        CONVERTED_ENCODER_DIR,
        dtype=torch.float32
    )
    
    print(f"  Original: {sum(p.numel() for p in original.parameters()):,} parameters")
    print(f"  Converted: {sum(p.numel() for p in converted.parameters()):,} parameters")
    
    return original, converted


def check_architecture_match(original, converted):
    """Verify that converted model has same architecture as original"""
    
    orig_state = original.state_dict()
    conv_state = converted.state_dict()
    
    orig_keys = set(orig_state.keys())
    conv_keys = set(conv_state.keys())
    
    # Check for matching keys
    missing_in_converted = orig_keys - conv_keys
    extra_in_converted = conv_keys - orig_keys
    common_keys = orig_keys & conv_keys
    
    print(f"\nTotal layers in original: {len(orig_keys)}")
    print(f"Total layers in converted: {len(conv_keys)}")
    print(f"Matching layers: {len(common_keys)}")
    
    if missing_in_converted:
        print(f"\nMissing in converted ({len(missing_in_converted)}):")
        for key in sorted(missing_in_converted)[:10]:
            print(f"  - {key}")
        if len(missing_in_converted) > 10:
            print(f"  ... and {len(missing_in_converted) - 10} more")
    
    if extra_in_converted:
        print(f"\nExtra in converted ({len(extra_in_converted)}):")
        for key in sorted(extra_in_converted)[:10]:
            print(f"  - {key}")
        if len(extra_in_converted) > 10:
            print(f"  ... and {len(extra_in_converted) - 10} more")
    
    # Check shape matching for common keys
    shape_mismatches = []
    for key in common_keys:
        if orig_state[key].shape != conv_state[key].shape:
            shape_mismatches.append((key, orig_state[key].shape, conv_state[key].shape))
    
    if shape_mismatches:
        print(f"\nShape mismatches ({len(shape_mismatches)}):")
        for key, orig_shape, conv_shape in shape_mismatches[:10]:
            print(f"  - {key}: {orig_shape} vs {conv_shape}")
        if len(shape_mismatches) > 10:
            print(f"  ... and {len(shape_mismatches) - 10} more")
    else:
        print("\nAll common layers have matching shapes")
    
    # Overall verdict
    if not missing_in_converted and not extra_in_converted and not shape_mismatches:
        print("\nARCHITECTURE MATCH: Perfect match!")
        return True
    else:
        print("\nARCHITECTURE MISMATCH: See details above")
        return False


def compute_frobenius_norms(original, converted):
    """Compute Frobenius norm (L2 distance) for each layer"""
    print("WEIGHT COMPARISON (Frobenius Norm / L2 Distance)")
    
    orig_state = original.state_dict()
    conv_state = converted.state_dict()
    
    common_keys = set(orig_state.keys()) & set(conv_state.keys())
    
    # Group by layer type
    layer_norms = {
        'embeddings': [],
        'attention': [],
        'mlp': [],
        'layer_norm': [],
        'projection': []
    }
    
    layer_details = []
    
    for key in sorted(common_keys):
        orig_weight = orig_state[key]
        conv_weight = conv_state[key]
        
        # Compute Frobenius norm (L2 distance)
        frobenius_norm = torch.norm(orig_weight - conv_weight, p='fro').item()
        
        layer_details.append({
            'name': key,
            'frobenius_norm': frobenius_norm
        })
        
        # Categorize
        if 'embedding' in key:
            layer_norms['embeddings'].append((key, frobenius_norm))
        elif 'attn' in key or 'attention' in key:
            layer_norms['attention'].append((key, frobenius_norm))
        elif 'mlp' in key:
            layer_norms['mlp'].append((key, frobenius_norm))
        elif 'layer_norm' in key:
            layer_norms['layer_norm'].append((key, frobenius_norm))
        elif 'projection' in key:
            layer_norms['projection'].append((key, frobenius_norm))
    
    # Print summary statistics
    print(f"\nTotal layers compared: {len(layer_details)}")
    
    all_norms = [d['frobenius_norm'] for d in layer_details]
    
    print(f"\nFrobenius Norm Statistics:")
    print(f"  Mean: {np.mean(all_norms):.6f}")
    print(f"  Median: {np.median(all_norms):.6f}")
    print(f"  Min: {np.min(all_norms):.6f}")
    print(f"  Max: {np.max(all_norms):.6f}")
    
    # Print top 10 layers with smallest differences
    print(f"\nTop 10 Layers with Smallest Differences:")
    sorted_details = sorted(layer_details, key=lambda x: x['frobenius_norm'], reverse=False)
    for i, detail in enumerate(sorted_details[:10], 1):
        print(f"  {i}. {detail['name']}")
        print(f"     Frobenius Norm: {detail['frobenius_norm']:.6f}")
    
    return layer_details, layer_norms


def create_visualization(layer_details, output_path=OUTPUT_PLOT):
    """Create visualization of weight differences"""
    print(f"\nGenerating visualization...")
    
    # Prepare data
    names = [d['name'] for d in layer_details]
    norms = [d['frobenius_norm'] for d in layer_details]
    
    # Shorten layer names for display
    short_names = []
    for name in names:
        parts = name.split('.')
        if 'encoder.layers' in name:
            # Format: "L{layer_num}.{component}"
            layer_num = parts[3]
            component = '.'.join(parts[4:])
            short_names.append(f"L{layer_num}.{component}")
        else:
            # Just take last 2 parts
            short_names.append('.'.join(parts[-2:]) if len(parts) > 1 else name)
    
    # Create single horizontal bar plot
    fig, ax = plt.subplots(figsize=(12, max(8, len(names) * 0.3)))
    
    y_pos = np.arange(len(short_names))
    ax.barh(y_pos, norms, alpha=0.7)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(short_names, fontsize=8)
    ax.set_xlabel('Frobenius Norm', fontsize=12)
    ax.set_ylabel('Layer', fontsize=12)
    ax.set_title('Weight Differences: CLIC vs SD3.5 Original', fontsize=14, fontweight='bold')
    ax.grid(alpha=0.3, axis='x')
    
    plt.tight_layout()
    plt.savefig(output_path, dpi=150, bbox_inches='tight')
    print(f"  Saved plot to: {output_path}")


def save_to_csv(layer_details, orig_state, conv_state, output_path=OUTPUT_CSV):
    """Save layer comparison data to CSV"""
    print(f"\nSaving data to CSV...")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['Layer Name', 'Frobenius Norm'])
        
        for detail in layer_details:
            name = detail['name']
            frobenius = detail['frobenius_norm']
            writer.writerow([name, f"{frobenius:.6f}"])
    print(f"  Saved CSV to: {output_path}")


def main():
    """Run full verification pipeline"""
    print("CONVERTED TEXT ENCODER VERIFICATION")
    
    original, converted = load_models()
    arch_match = check_architecture_match(original, converted)
    layer_details, layer_norms = compute_frobenius_norms(original, converted)
    
    # Save to CSV
    save_to_csv(layer_details, original.state_dict(), converted.state_dict())
    create_visualization(layer_details)

    if arch_match:
        print("Architecture matches SD3.5 requirements")
    else:
        print("Architecture has differences from SD3.5")
    
    all_norms = [d['frobenius_norm'] for d in layer_details]
    if all(norm > 1e6 for norm in all_norms):
        print("All weights are different (all norms > 1e6)")
    elif any(norm > 1e6 for norm in all_norms):
        print("Some weights are different (some norms > 1e6)")
    else:
        print("Weights have moderate differences (all norms <= 1e6)")
    print(f"\nAverage Frobenius norm: {np.mean(all_norms):.6f}")


if __name__ == "__main__":
    main()
