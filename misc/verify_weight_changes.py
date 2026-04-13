"""
Verify Weight Changes After Conversion
=======================================
Compare weights between:
- Original OpenCLIP model
- Converted Hugging Face model
- Original SD3.5 text_encoder

Most weights should be different after conversion.
"""

import torch
import open_clip
from transformers import CLIPTextModelWithProjection
import os

# Paths
OPENCLIP_MODEL = "hf-hub:nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps"
CONVERTED_PATH = "converted_text_encoder"
SD35_PATH = "stable-diffusion-3.5-large-turbo"

print("=" * 80)
print("WEIGHT CHANGE VERIFICATION")
print("=" * 80)

# Load models
print("\n[1/3] Loading OpenCLIP model...")
openclip_model, _, _ = open_clip.create_model_and_transforms(
    OPENCLIP_MODEL,
    precision='fp32',
    device='cpu'
)
openclip_state = openclip_model.state_dict()
print(f"  Loaded {len(openclip_state)} parameters")

print("\n[2/3] Loading converted Hugging Face model...")
if not os.path.exists(CONVERTED_PATH):
    print(f"ERROR: Converted model not found at {CONVERTED_PATH}")
    print("Run the conversion first!")
    exit(1)
    
converted_model = CLIPTextModelWithProjection.from_pretrained(
    CONVERTED_PATH,
    torch_dtype=torch.float32,
    device_map='cpu'
)
converted_state = converted_model.state_dict()
print(f"  Loaded {len(converted_state)} parameters")

print("\n[3/3] Loading original SD3.5 text_encoder...")
sd35_encoder = CLIPTextModelWithProjection.from_pretrained(
    SD35_PATH,
    subfolder="text_encoder",
    torch_dtype=torch.float32,
    device_map='cpu'
)
sd35_state = sd35_encoder.state_dict()
print(f"  Loaded {len(sd35_state)} parameters")

# Compare shapes
print("\n" + "=" * 80)
print("SHAPE COMPARISON")
print("=" * 80)

print("\nConverted vs OpenCLIP (should have similar total params):")
conv_params = sum(p.numel() for p in converted_model.parameters())
openclip_params = sum(p.numel() for p in openclip_model.parameters())
print(f"  Converted:  {conv_params:,} parameters")
print(f"  OpenCLIP:   {openclip_params:,} parameters")

print("\nConverted vs SD3.5 text_encoder (should be same architecture):")
sd35_params = sum(p.numel() for p in sd35_encoder.parameters())
print(f"  Converted:  {conv_params:,} parameters")
print(f"  SD3.5:      {sd35_params:,} parameters")
if conv_params == sd35_params:
    print("  ✓ Architecture matches!")
else:
    print("  ✗ Architecture mismatch!")

# Weight change analysis
print("\n" + "=" * 80)
print("WEIGHT CHANGE ANALYSIS")
print("=" * 80)

print("\n[A] Converted vs SD3.5 Original (should be DIFFERENT):")
print("-" * 80)

identical_to_sd35 = []
different_from_sd35 = []
only_in_converted = []
only_in_sd35 = []

for key in converted_state.keys():
    if key not in sd35_state:
        only_in_converted.append(key)
        continue
    
    conv_tensor = converted_state[key]
    sd35_tensor = sd35_state[key]
    
    if conv_tensor.shape != sd35_tensor.shape:
        different_from_sd35.append((key, "shape_mismatch"))
        continue
    
    max_diff = torch.abs(conv_tensor - sd35_tensor).max().item()
    
    if max_diff < 1e-6:
        identical_to_sd35.append(key)
    else:
        different_from_sd35.append((key, max_diff))

for key in sd35_state.keys():
    if key not in converted_state:
        only_in_sd35.append(key)

print(f"\nIdentical weights (UNEXPECTED): {len(identical_to_sd35)}")
if identical_to_sd35:
    for key in identical_to_sd35[:5]:
        print(f"  - {key}")
    if len(identical_to_sd35) > 5:
        print(f"  ... and {len(identical_to_sd35) - 5} more")

print(f"\nChanged weights (EXPECTED): {len(different_from_sd35)}")
for key, diff in different_from_sd35[:10]:
    if diff == "shape_mismatch":
        print(f"  - {key}: SHAPE MISMATCH")
    else:
        print(f"  - {key}: max_diff={diff:.6f}")
if len(different_from_sd35) > 10:
    print(f"  ... and {len(different_from_sd35) - 10} more")

if only_in_converted:
    print(f"\nOnly in converted: {len(only_in_converted)}")
    for key in only_in_converted[:3]:
        print(f"  - {key}")

if only_in_sd35:
    print(f"\nOnly in SD3.5: {len(only_in_sd35)}")
    for key in only_in_sd35[:3]:
        print(f"  - {key}")

# Sample weight values
print("\n" + "=" * 80)
print("SAMPLE WEIGHT VALUES")
print("=" * 80)

sample_keys = [
    "text_model.embeddings.token_embedding.weight",
    "text_model.encoder.layers.0.self_attn.q_proj.weight",
    "text_model.encoder.layers.0.mlp.fc1.weight",
    "text_projection.weight"
]

for key in sample_keys:
    print(f"\n{key}:")
    
    if key in converted_state:
        conv_val = converted_state[key].flatten()[:5]
        print(f"  Converted: {conv_val.tolist()}")
    else:
        print(f"  Converted: NOT FOUND")
    
    if key in sd35_state:
        sd35_val = sd35_state[key].flatten()[:5]
        print(f"  SD3.5:     {sd35_val.tolist()}")
    else:
        print(f"  SD3.5:     NOT FOUND")
    
    if key in converted_state and key in sd35_state:
        diff = torch.abs(converted_state[key] - sd35_state[key]).max().item()
        print(f"  Max Diff:  {diff:.6f}")

# Final verdict
print("\n" + "=" * 80)
print("VERIFICATION RESULT")
print("=" * 80)

total_weights = len(converted_state)
changed_count = len(different_from_sd35)
unchanged_count = len(identical_to_sd35)

print(f"\nTotal weights:     {total_weights}")
print(f"Changed:           {changed_count} ({100*changed_count/total_weights:.1f}%)")
print(f"Unchanged:         {unchanged_count} ({100*unchanged_count/total_weights:.1f}%)")

if changed_count > total_weights * 0.9:
    print("\n✓ SUCCESS: Most weights have changed (conversion worked!)")
elif changed_count > total_weights * 0.5:
    print("\n⚠ PARTIAL: Some weights changed, but not enough")
else:
    print("\n✗ FAILED: Most weights are still the same (conversion didn't work)")

print("\n" + "=" * 80)

