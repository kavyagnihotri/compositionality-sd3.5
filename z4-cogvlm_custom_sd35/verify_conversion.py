"""
Checks:
1. Token embeddings 
2. Positional embeddings 
3. QKV split correctness (fused -> separate Q, K, V)
4. MLP layer renaming (c_fc -> fc1, c_proj -> fc2)
5. Text projection transpose
6. Layer norms
"""

import torch
from pathlib import Path
from transformers import CLIPTextModelWithProjection
import open_clip
import matplotlib.pyplot as plt
import numpy as np
import csv


# Configuration
OPENCLIP_CHECKPOINT = 'hf-hub:nmndeep/CLIC-ViT-L-14-224-CogVLM'
CONVERTED_ENCODER_DIR = "./converted_text_encoder_CogVLM"
OUTPUT_PLOT = "./conversion_verification.png"
OUTPUT_CSV = "./conversion_verification.csv"
NUM_LAYERS = 12  # Number of layers we converted


def load_models():
    """Load original OpenCLIP and converted HuggingFace models"""
    print("Loading models...")
    
    # Load OpenCLIP
    print("  Loading original CLIC OpenCLIP model (CogVLM)...")
    openclip_model, _, _ = open_clip.create_model_and_transforms(OPENCLIP_CHECKPOINT)
    openclip_model.eval()
    
    # Load converted HuggingFace
    print("  Loading converted HuggingFace model...")
    hf_model = CLIPTextModelWithProjection.from_pretrained(
        CONVERTED_ENCODER_DIR,
        dtype=torch.float32
    )
    hf_model.eval()
    
    return openclip_model, hf_model


def check_unmapped_weights(hf_state, conversion_script_path='./convert_text_encoder.py'):
    """Identify weights in HF model that were not mapped from CLIC"""
    print("\nCHECKING FOR UNMAPPED WEIGHTS")
    
    # Get list of mapped weight keys from the conversion logic
    mapped_keys = set()
    
    # Embeddings
    mapped_keys.add('text_model.embeddings.token_embedding.weight')
    mapped_keys.add('text_model.embeddings.position_embedding.weight')
    
    # Transformer layers (only first NUM_LAYERS)
    for layer_idx in range(NUM_LAYERS):
        # Attention
        mapped_keys.update([
            f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight',
            f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias',
            f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight',
            f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias',
            f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight',
            f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias',
            f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight',
            f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias',
        ])
        
        # Layer norms
        mapped_keys.update([
            f'text_model.encoder.layers.{layer_idx}.layer_norm1.weight',
            f'text_model.encoder.layers.{layer_idx}.layer_norm1.bias',
            f'text_model.encoder.layers.{layer_idx}.layer_norm2.weight',
            f'text_model.encoder.layers.{layer_idx}.layer_norm2.bias',
        ])
        
        # MLP
        mapped_keys.update([
            f'text_model.encoder.layers.{layer_idx}.mlp.fc1.weight',
            f'text_model.encoder.layers.{layer_idx}.mlp.fc1.bias',
            f'text_model.encoder.layers.{layer_idx}.mlp.fc2.weight',
            f'text_model.encoder.layers.{layer_idx}.mlp.fc2.bias',
        ])
    
    # Final layers
    mapped_keys.add('text_model.final_layer_norm.weight')
    mapped_keys.add('text_model.final_layer_norm.bias')
    mapped_keys.add('text_projection.weight')
    
    # Find unmapped keys
    unmapped_keys = []
    for key in hf_state.keys():
        if key not in mapped_keys:
            unmapped_keys.append(key)
    
    if unmapped_keys:
        print(f"\nWARNING: {len(unmapped_keys)} weights in HF model were NOT mapped from CLIC:")
        print("These weights remain at their RANDOM INITIALIZATION values!\n")
        for key in sorted(unmapped_keys):
            tensor = hf_state[key]
            print(f"  - {key}")
            print(f"    Shape: {tuple(tensor.shape)}, Mean: {tensor.mean().item():.6f}, Std: {tensor.std().item():.6f}")
    else:
        print("\nAll HF model weights were successfully mapped from CLIC")
    
    return unmapped_keys


def verify_embeddings(oc_state, hf_state):
    """Verify token and positional embeddings were preserved"""
    print("EMBEDDINGS VERIFICATION")
    
    results = []
    
    # Token embeddings
    oc_token = oc_state['token_embedding.weight']
    hf_token = hf_state['text_model.embeddings.token_embedding.weight']
    diff = torch.norm(oc_token - hf_token, p='fro').item()
    max_diff = (oc_token - hf_token).abs().max().item()
    
    print(f"\nToken Embeddings:")
    print(f"  Shape: {oc_token.shape}")
    print(f"  Frobenius norm diff: {diff:.2e}")
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Status: {' PASS' if diff < 1e-6 else ' FAIL'}")
    
    results.append({
        'oc_name': 'token_embedding.weight',
        'hf_name': 'text_model.embeddings.token_embedding.weight',
        'name': 'token_embedding',
        'frobenius': diff,
        'max_abs': max_diff,
        'pass': diff < 1e-6
    })
    
    # Positional embeddings
    oc_pos = oc_state['positional_embedding']
    hf_pos = hf_state['text_model.embeddings.position_embedding.weight']
    diff = torch.norm(oc_pos - hf_pos, p='fro').item()
    max_diff = (oc_pos - hf_pos).abs().max().item()
    
    print(f"\nPositional Embeddings:")
    print(f"  Shape: {oc_pos.shape}")
    print(f"  Frobenius norm diff: {diff:.2e}")
    print(f"  Max absolute diff: {max_diff:.2e}")
    print(f"  Status: {' PASS' if diff < 1e-6 else ' FAIL'}")
    
    results.append({
        'oc_name': 'positional_embedding',
        'hf_name': 'text_model.embeddings.position_embedding.weight',
        'name': 'positional_embedding',
        'frobenius': diff,
        'max_abs': max_diff,
        'pass': diff < 1e-6
    })
    
    return results


def verify_qkv_split(oc_state, hf_state, layer_idx):
    """Verify QKV attention split for a specific layer"""
    oc_qkv_weight = oc_state[f'transformer.resblocks.{layer_idx}.attn.in_proj_weight']
    oc_qkv_bias = oc_state[f'transformer.resblocks.{layer_idx}.attn.in_proj_bias']
    
    chunk_size = oc_qkv_weight.shape[0] // 3
    
    # Split original QKV
    oc_q_w = oc_qkv_weight[:chunk_size, :]
    oc_k_w = oc_qkv_weight[chunk_size:2*chunk_size, :]
    oc_v_w = oc_qkv_weight[2*chunk_size:, :]
    
    oc_q_b = oc_qkv_bias[:chunk_size]
    oc_k_b = oc_qkv_bias[chunk_size:2*chunk_size]
    oc_v_b = oc_qkv_bias[2*chunk_size:]
    
    # Get converted separate Q, K, V
    hf_q_w = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight']
    hf_k_w = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight']
    hf_v_w = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight']
    
    hf_q_b = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias']
    hf_k_b = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias']
    hf_v_b = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias']
    
    # Compare
    results = []
    
    for name, oc_tensor, hf_tensor, oc_key, hf_key in [
        ('Q_weight', oc_q_w, hf_q_w, f'transformer.resblocks.{layer_idx}.attn.in_proj_weight (Q)', f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.weight'),
        ('K_weight', oc_k_w, hf_k_w, f'transformer.resblocks.{layer_idx}.attn.in_proj_weight (K)', f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.weight'),
        ('V_weight', oc_v_w, hf_v_w, f'transformer.resblocks.{layer_idx}.attn.in_proj_weight (V)', f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.weight'),
        ('Q_bias', oc_q_b, hf_q_b, f'transformer.resblocks.{layer_idx}.attn.in_proj_bias (Q)', f'text_model.encoder.layers.{layer_idx}.self_attn.q_proj.bias'),
        ('K_bias', oc_k_b, hf_k_b, f'transformer.resblocks.{layer_idx}.attn.in_proj_bias (K)', f'text_model.encoder.layers.{layer_idx}.self_attn.k_proj.bias'),
        ('V_bias', oc_v_b, hf_v_b, f'transformer.resblocks.{layer_idx}.attn.in_proj_bias (V)', f'text_model.encoder.layers.{layer_idx}.self_attn.v_proj.bias'),
    ]:
        diff = torch.norm(oc_tensor - hf_tensor, p='fro').item()
        max_diff = (oc_tensor - hf_tensor).abs().max().item()
        
        results.append({
            'oc_name': oc_key,
            'hf_name': hf_key,
            'layer': layer_idx,
            'component': name,
            'frobenius': diff,
            'max_abs': max_diff,
            'pass': diff < 1e-6
        })
    
    return results


def verify_mlp_layers(oc_state, hf_state, layer_idx):
    """Verify MLP layer renaming (c_fc -> fc1, c_proj -> fc2)"""
    results = []
    
    # c_fc -> fc1
    oc_fc1_w = oc_state[f'transformer.resblocks.{layer_idx}.mlp.c_fc.weight']
    oc_fc1_b = oc_state[f'transformer.resblocks.{layer_idx}.mlp.c_fc.bias']
    hf_fc1_w = hf_state[f'text_model.encoder.layers.{layer_idx}.mlp.fc1.weight']
    hf_fc1_b = hf_state[f'text_model.encoder.layers.{layer_idx}.mlp.fc1.bias']
    
    diff_w = torch.norm(oc_fc1_w - hf_fc1_w, p='fro').item()
    diff_b = torch.norm(oc_fc1_b - hf_fc1_b, p='fro').item()
    
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.mlp.c_fc.weight',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.mlp.fc1.weight',
        'layer': layer_idx,
        'component': 'fc1_weight',
        'frobenius': diff_w,
        'max_abs': (oc_fc1_w - hf_fc1_w).abs().max().item(),
        'pass': diff_w < 1e-6
    })
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.mlp.c_fc.bias',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.mlp.fc1.bias',
        'layer': layer_idx,
        'component': 'fc1_bias',
        'frobenius': diff_b,
        'max_abs': (oc_fc1_b - hf_fc1_b).abs().max().item(),
        'pass': diff_b < 1e-6
    })
    
    # c_proj -> fc2
    oc_fc2_w = oc_state[f'transformer.resblocks.{layer_idx}.mlp.c_proj.weight']
    oc_fc2_b = oc_state[f'transformer.resblocks.{layer_idx}.mlp.c_proj.bias']
    hf_fc2_w = hf_state[f'text_model.encoder.layers.{layer_idx}.mlp.fc2.weight']
    hf_fc2_b = hf_state[f'text_model.encoder.layers.{layer_idx}.mlp.fc2.bias']
    
    diff_w = torch.norm(oc_fc2_w - hf_fc2_w, p='fro').item()
    diff_b = torch.norm(oc_fc2_b - hf_fc2_b, p='fro').item()
    
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.mlp.c_proj.weight',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.mlp.fc2.weight',
        'layer': layer_idx,
        'component': 'fc2_weight',
        'frobenius': diff_w,
        'max_abs': (oc_fc2_w - hf_fc2_w).abs().max().item(),
        'pass': diff_w < 1e-6
    })
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.mlp.c_proj.bias',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.mlp.fc2.bias',
        'layer': layer_idx,
        'component': 'fc2_bias',
        'frobenius': diff_b,
        'max_abs': (oc_fc2_b - hf_fc2_b).abs().max().item(),
        'pass': diff_b < 1e-6
    })
    
    return results


def verify_attention_out_proj(oc_state, hf_state, layer_idx):
    """Verify attention output projection"""
    oc_out_w = oc_state[f'transformer.resblocks.{layer_idx}.attn.out_proj.weight']
    oc_out_b = oc_state[f'transformer.resblocks.{layer_idx}.attn.out_proj.bias']
    hf_out_w = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight']
    hf_out_b = hf_state[f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias']
    
    diff_w = torch.norm(oc_out_w - hf_out_w, p='fro').item()
    diff_b = torch.norm(oc_out_b - hf_out_b, p='fro').item()
    
    return [
        {
            'oc_name': f'transformer.resblocks.{layer_idx}.attn.out_proj.weight',
            'hf_name': f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.weight',
            'layer': layer_idx,
            'component': 'attn_out_weight',
            'frobenius': diff_w,
            'max_abs': (oc_out_w - hf_out_w).abs().max().item(),
            'pass': diff_w < 1e-6
        },
        {
            'oc_name': f'transformer.resblocks.{layer_idx}.attn.out_proj.bias',
            'hf_name': f'text_model.encoder.layers.{layer_idx}.self_attn.out_proj.bias',
            'layer': layer_idx,
            'component': 'attn_out_bias',
            'frobenius': diff_b,
            'max_abs': (oc_out_b - hf_out_b).abs().max().item(),
            'pass': diff_b < 1e-6
        }
    ]


def verify_layer_norms(oc_state, hf_state, layer_idx):
    """Verify layer normalization weights"""
    results = []
    
    # LayerNorm 1
    oc_ln1_w = oc_state[f'transformer.resblocks.{layer_idx}.ln_1.weight']
    oc_ln1_b = oc_state[f'transformer.resblocks.{layer_idx}.ln_1.bias']
    hf_ln1_w = hf_state[f'text_model.encoder.layers.{layer_idx}.layer_norm1.weight']
    hf_ln1_b = hf_state[f'text_model.encoder.layers.{layer_idx}.layer_norm1.bias']
    
    diff_w = torch.norm(oc_ln1_w - hf_ln1_w, p='fro').item()
    diff_b = torch.norm(oc_ln1_b - hf_ln1_b, p='fro').item()
    
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.ln_1.weight',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.layer_norm1.weight',
        'layer': layer_idx,
        'component': 'ln1_weight',
        'frobenius': diff_w,
        'max_abs': (oc_ln1_w - hf_ln1_w).abs().max().item(),
        'pass': diff_w < 1e-6
    })
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.ln_1.bias',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.layer_norm1.bias',
        'layer': layer_idx,
        'component': 'ln1_bias',
        'frobenius': diff_b,
        'max_abs': (oc_ln1_b - hf_ln1_b).abs().max().item(),
        'pass': diff_b < 1e-6
    })
    
    # LayerNorm 2
    oc_ln2_w = oc_state[f'transformer.resblocks.{layer_idx}.ln_2.weight']
    oc_ln2_b = oc_state[f'transformer.resblocks.{layer_idx}.ln_2.bias']
    hf_ln2_w = hf_state[f'text_model.encoder.layers.{layer_idx}.layer_norm2.weight']
    hf_ln2_b = hf_state[f'text_model.encoder.layers.{layer_idx}.layer_norm2.bias']
    
    diff_w = torch.norm(oc_ln2_w - hf_ln2_w, p='fro').item()
    diff_b = torch.norm(oc_ln2_b - hf_ln2_b, p='fro').item()
    
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.ln_2.weight',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.layer_norm2.weight',
        'layer': layer_idx,
        'component': 'ln2_weight',
        'frobenius': diff_w,
        'max_abs': (oc_ln2_w - hf_ln2_w).abs().max().item(),
        'pass': diff_w < 1e-6
    })
    results.append({
        'oc_name': f'transformer.resblocks.{layer_idx}.ln_2.bias',
        'hf_name': f'text_model.encoder.layers.{layer_idx}.layer_norm2.bias',
        'layer': layer_idx,
        'component': 'ln2_bias',
        'frobenius': diff_b,
        'max_abs': (oc_ln2_b - hf_ln2_b).abs().max().item(),
        'pass': diff_b < 1e-6
    })
    
    return results


def verify_final_layers(oc_state, hf_state):
    """Verify final layer norm and text projection"""
    results = []
    
    # Final layer norm
    oc_ln_w = oc_state['ln_final.weight']
    oc_ln_b = oc_state['ln_final.bias']
    hf_ln_w = hf_state['text_model.final_layer_norm.weight']
    hf_ln_b = hf_state['text_model.final_layer_norm.bias']
    
    diff_w = torch.norm(oc_ln_w - hf_ln_w, p='fro').item()
    diff_b = torch.norm(oc_ln_b - hf_ln_b, p='fro').item()
    
    print(f"\nFinal Layer Norm:")
    print(f"  Weight diff: {diff_w:.2e}, Max abs: {(oc_ln_w - hf_ln_w).abs().max().item():.2e}")
    print(f"  Bias diff: {diff_b:.2e}, Max abs: {(oc_ln_b - hf_ln_b).abs().max().item():.2e}")
    print(f"  Status: {' PASS' if diff_w < 1e-6 and diff_b < 1e-6 else ' FAIL'}")
    
    results.append({
        'oc_name': 'ln_final.weight',
        'hf_name': 'text_model.final_layer_norm.weight',
        'name': 'ln_final_weight',
        'frobenius': diff_w,
        'max_abs': (oc_ln_w - hf_ln_w).abs().max().item(),
        'pass': diff_w < 1e-6
    })
    results.append({
        'oc_name': 'ln_final.bias',
        'hf_name': 'text_model.final_layer_norm.bias',
        'name': 'ln_final_bias',
        'frobenius': diff_b,
        'max_abs': (oc_ln_b - hf_ln_b).abs().max().item(),
        'pass': diff_b < 1e-6
    })
    
    # Text projection (TRANSPOSED)
    if 'text_projection' in oc_state:
        oc_proj = oc_state['text_projection'].T  # Transpose for comparison
        hf_proj = hf_state['text_projection.weight']
        
        diff = torch.norm(oc_proj - hf_proj, p='fro').item()
        max_diff = (oc_proj - hf_proj).abs().max().item()
        
        print(f"\nText Projection (Transposed):")
        print(f"  OpenCLIP shape: {oc_state['text_projection'].shape} -> {oc_proj.shape} (after transpose)")
        print(f"  HuggingFace shape: {hf_proj.shape}")
        print(f"  Frobenius norm diff: {diff:.2e}")
        print(f"  Max absolute diff: {max_diff:.2e}")
        print(f"  Status: {' PASS' if diff < 1e-6 else ' FAIL'}")
        
        results.append({
            'oc_name': 'text_projection',
            'hf_name': 'text_projection.weight',
            'name': 'text_projection',
            'frobenius': diff,
            'max_abs': max_diff,
            'pass': diff < 1e-6
        })
    
    return results


def verify_all_layers(oc_state, hf_state):
    """Verify all transformer layers"""
    print("TRANSFORMER LAYERS VERIFICATION")
    
    all_results = []
    
    for layer_idx in range(NUM_LAYERS):
        print(f"\nLayer {layer_idx}:")
        
        # QKV split
        qkv_results = verify_qkv_split(oc_state, hf_state, layer_idx)
        qkv_pass = all(r['pass'] for r in qkv_results)
        max_qkv_diff = max(r['max_abs'] for r in qkv_results)
        print(f"  QKV split: {'' if qkv_pass else ''} (max diff: {max_qkv_diff:.2e})")
        all_results.extend(qkv_results)
        
        # Attention out projection
        attn_out_results = verify_attention_out_proj(oc_state, hf_state, layer_idx)
        attn_out_pass = all(r['pass'] for r in attn_out_results)
        max_attn_out_diff = max(r['max_abs'] for r in attn_out_results)
        print(f"  Attention out: {'' if attn_out_pass else ''} (max diff: {max_attn_out_diff:.2e})")
        all_results.extend(attn_out_results)
        
        # MLP layers
        mlp_results = verify_mlp_layers(oc_state, hf_state, layer_idx)
        mlp_pass = all(r['pass'] for r in mlp_results)
        max_mlp_diff = max(r['max_abs'] for r in mlp_results)
        print(f"  MLP (fc1/fc2): {'' if mlp_pass else ''} (max diff: {max_mlp_diff:.2e})")
        all_results.extend(mlp_results)
        
        # Layer norms
        ln_results = verify_layer_norms(oc_state, hf_state, layer_idx)
        ln_pass = all(r['pass'] for r in ln_results)
        max_ln_diff = max(r['max_abs'] for r in ln_results)
        print(f"  Layer norms: {'' if ln_pass else ''} (max diff: {max_ln_diff:.2e})")
        all_results.extend(ln_results)
    
    return all_results


def save_results_to_csv(all_results, output_path):
    """Save verification results to CSV"""
    print(f"\nSaving results to CSV...")
    
    with open(output_path, 'w', newline='') as f:
        writer = csv.writer(f)
        writer.writerow(['CLIC Name', 'Converted SD3.5 Name', 'Frobenius Norm', 'Max Absolute Diff'])
        
        for result in all_results:
            oc_name = result.get('oc_name', 'N/A')
            hf_name = result.get('hf_name') or result.get('name') or f"Layer {result['layer']} - {result['component']}"
            frobenius = result['frobenius']
            writer.writerow([oc_name, hf_name, f"{frobenius:.2e}"])
    
    print(f"  Saved to: {output_path}")


def main():
    """Run full conversion verification"""
    print("CLIC COGVLM OPENCLIP TO HUGGINGFACE CONVERSION VERIFICATION\n")
    
    # Load models
    openclip_model, hf_model = load_models()
    
    oc_state = openclip_model.state_dict()
    hf_state = hf_model.state_dict()
    
    # Check for unmapped weights first
    unmapped_keys = check_unmapped_weights(hf_state)
    
    # Verify embeddings
    emb_results = verify_embeddings(oc_state, hf_state)
    
    # Verify all transformer layers
    layer_results = verify_all_layers(oc_state, hf_state)
    
    # Verify final layers
    print("FINAL LAYERS VERIFICATION")
    final_results = verify_final_layers(oc_state, hf_state)
    
    # Combine all results
    all_results = emb_results + layer_results + final_results
    
    # Save to CSV
    save_results_to_csv(all_results, OUTPUT_CSV)
    
    # Final summary
    print("CONVERSION VERIFICATION COMPLETE")
    
    total_checks = len(all_results)
    passed_checks = sum(1 for r in all_results if r['pass'])
    failed_checks = total_checks - passed_checks
    
    print(f"\nMapped Weight Verification:")
    print(f"  Total checks: {total_checks}")
    print(f"  Passed: {passed_checks}")
    print(f"  Failed: {failed_checks}")
    print(f"  Success rate: {100 * passed_checks / total_checks:.2f}%")
    
    print(f"\nUnmapped Weights: {len(unmapped_keys)}")
    if unmapped_keys:
        print("  These weights remain RANDOMLY INITIALIZED (not from CLIC)")
    
    if failed_checks == 0 and len(unmapped_keys) == 0:
        print("\nALL CHECKS PASSED Conversion was performed correctly!")
        print("All weights successfully mapped from CLIC CogVLM")
    elif failed_checks == 0:
        print("\nAll mapped weights verified correctly")
        print(f"BUT {len(unmapped_keys)} weights are unmapped (random initialization)")
    else:
        print(f"\n{failed_checks} checks failed Review failed components")
        print("\nFailed components:")
        for r in all_results:
            if not r['pass']:
                name = r.get('name') or f"Layer {r['layer']} - {r['component']}"
                print(f"  - {name}: max_abs={r['max_abs']:.2e}, frobenius={r['frobenius']:.2e}")


if __name__ == "__main__":
    main()
