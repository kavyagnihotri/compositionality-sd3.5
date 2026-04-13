# Replacing SD3.5's Text Encoder with a Custom OpenCLIP Model (CLIC)

## Table of Contents

1. [Overview & Architecture](#1-overview--architecture)
2. [Phase 1 — Weight Conversion ("Weight Surgery")](#2-phase-1--weight-conversion-weight-surgery)
3. [Phase 2 — Tokenizer Migration](#3-phase-2--tokenizer-migration)
4. [Phase 3 — Verification](#4-phase-3--verification)
5. [Phase 4 — Component-Based Pipeline Integration](#5-phase-4--component-based-pipeline-integration)
6. [Phase 5 — Fine-Tuning Strategy](#6-phase-5--fine-tuning-strategy)
7. [Known Issues & Gaps in Current Code](#7-known-issues--gaps-in-current-code)
8. [Quick Reference: File Map](#8-quick-reference-file-map)

---

## 1. Overview & Architecture

### What SD3.5 Large Turbo has inside

Stable Diffusion 3.5 uses **three** text encoders in parallel. Each one converts your text prompt into a vector representation that guides image generation:

| Component        | Architecture       | Hidden Dim | Role                          |
|------------------|--------------------|------------|-------------------------------|
| `text_encoder`   | CLIP-L/14 (768-d)  | 768        | Short-range semantics         |
| `text_encoder_2` | CLIP-G/14 (1280-d) | 1280       | Broader visual-text alignment |
| `text_encoder_3` | T5-XXL (4096-d)    | 4096       | Rich language understanding   |

The pipeline concatenates all three encoders' outputs and feeds them to the **MMDiT** (Multi-Modal Diffusion Transformer) backbone, which generates the image.

### What we are doing

We **replace only `text_encoder`** (CLIP-L) with a custom-trained OpenCLIP model called **CLIC** (`nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps`). The other two encoders (`text_encoder_2`, `text_encoder_3`) stay untouched.

### Why this is non-trivial

The custom CLIC model is saved in **OpenCLIP format** (a single `.safetensors` with fused QKV attention, different layer names, transposed projection). SD3.5 expects **Hugging Face `transformers` format** (separated Q/K/V weights, `CLIPTextModelWithProjection` class). You cannot just swap the file — you need a conversion bridge.

---

## 2. Phase 1 — Weight Conversion ("Weight Surgery")

> **File:** `custom_sd35.py` → `convert_openclip_to_transformers()`

### Why

OpenCLIP and Hugging Face store the exact same neural network with different conventions. Think of it like the same book written in two different languages — the content is identical but the encoding is different.

### What the code does

#### Step 1: Detect architecture from weights

```python
state_dict = openclip_model.state_dict()

text_width = state_dict['token_embedding.weight'].shape[1]    # → 768
vocab_size = state_dict['token_embedding.weight'].shape[0]    # → 49408
context_length = state_dict['positional_embedding'].shape[0]  # → 77
```

We read the shapes directly from the weight tensors. For CLIC-ViT-L-14 this gives us:
- `hidden_size=768` (12 heads × 64 dims each)
- `vocab_size=49408` (standard BPE vocabulary)
- `context_length=77` (max tokens per prompt)

#### Step 2: Create the HF config with QuickGELU

```python
config = CLIPTextConfig(
    hidden_size=768,
    intermediate_size=3072,       # 768 × 4
    num_hidden_layers=12,
    num_attention_heads=12,       # 768 / 64
    max_position_embeddings=77,
    vocab_size=49408,
    projection_dim=768,
    hidden_act="quick_gelu",      # ← CRITICAL
)
```

**Why QuickGELU matters:** OpenCLIP (and original OpenAI CLIP) uses `QuickGELU`: $x \cdot \sigma(1.702x)$. Standard HF CLIP uses regular `GELU`: $x \cdot \Phi(x)$. If you get this wrong, every single MLP output in all 12 layers will be numerically different, and the final embeddings will be garbage. The original SD3.5 `text_encoder` also uses `quick_gelu` (check its `config.json`), so this is consistent.

#### Step 3: QKV Splitting (the hardest part)

OpenCLIP fuses Query, Key, and Value into one giant matrix for efficiency:

```
OpenCLIP:   in_proj_weight  shape = [2304, 768]   (3 × 768 = 2304)
            in_proj_bias    shape = [2304]

HuggingFace: q_proj.weight  shape = [768, 768]    ← rows 0..767
             k_proj.weight  shape = [768, 768]    ← rows 768..1535
             v_proj.weight  shape = [768, 768]    ← rows 1536..2303
```

The code splits them:

```python
if '.attn.in_proj_weight' in k:
    chunk_size = v.shape[0] // 3                        # 2304 / 3 = 768
    new_sd[f'...q_proj.weight'] = v[:chunk_size, :]     # first third  → Q
    new_sd[f'...k_proj.weight'] = v[chunk_size:2*chunk_size, :]  # middle → K
    new_sd[f'...v_proj.weight'] = v[2*chunk_size:, :]   # last third  → V
```

**Order matters!** OpenCLIP always stores them as Q, K, V (in that order). If you mix up the order, attention will attend to wrong positions and outputs will be nonsensical.

#### Step 4: MLP layer renaming

```
OpenCLIP name    →    HuggingFace name
─────────────────────────────────────
mlp.c_fc.weight  →    mlp.fc1.weight      (first linear layer)
mlp.c_fc.bias    →    mlp.fc1.bias
mlp.c_proj.weight →   mlp.fc2.weight      (second linear layer)
mlp.c_proj.bias  →    mlp.fc2.bias
```

Pure renaming — no data transformation. The matrices are identical, just stored under different dictionary keys.

#### Step 5: Text projection transpose

```python
elif k == 'text_projection':
    new_state_dict['text_projection.weight'] = v.T   # TRANSPOSE!
```

**Why transpose?** OpenCLIP stores the text projection as a raw `nn.Parameter` with shape `[hidden_size, projection_dim]` (i.e., `[768, 768]`). HuggingFace wraps it in an `nn.Linear`, which stores weights transposed: `[projection_dim, hidden_size]`. For a square matrix this doesn't change the shape, but the values' row/column meaning flips. Forgetting this makes the final pooled embedding point in a completely wrong direction.

#### Step 6: Embedding remapping

```
token_embedding.weight        →  text_model.embeddings.token_embedding.weight
positional_embedding          →  text_model.embeddings.position_embedding.weight
ln_final.weight               →  text_model.final_layer_norm.weight
ln_final.bias                 →  text_model.final_layer_norm.bias
```

#### Step 7: Manual QuickGELU patching

```python
class QuickGELU(nn.Module):
    def forward(self, x):
        return x * torch.sigmoid(1.702 * x)

def patch_quickgelu_activation(model):
    for layer in model.text_model.encoder.layers:
        layer.mlp.activation_fn = QuickGELU()
    return model
```

**Why manual patching?** Setting `hidden_act="quick_gelu"` in the config is necessary but **not sufficient** in all HF `transformers` versions. Some versions silently fall back to GELU. The manual patch guarantees correctness regardless of library version. This is applied:
- Immediately after conversion (in `convert_openclip_to_transformers`)
- After loading from disk (in `create_sd35_with_custom_encoder`)

### Complete key mapping table

| OpenCLIP Key | HuggingFace Key | Transform |
|---|---|---|
| `token_embedding.weight` | `text_model.embeddings.token_embedding.weight` | Copy |
| `positional_embedding` | `text_model.embeddings.position_embedding.weight` | Copy |
| `transformer.resblocks.{i}.attn.in_proj_weight` | `text_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.weight` | Split into 3 |
| `transformer.resblocks.{i}.attn.in_proj_bias` | `text_model.encoder.layers.{i}.self_attn.{q,k,v}_proj.bias` | Split into 3 |
| `transformer.resblocks.{i}.attn.out_proj.weight` | `text_model.encoder.layers.{i}.self_attn.out_proj.weight` | Copy |
| `transformer.resblocks.{i}.attn.out_proj.bias` | `text_model.encoder.layers.{i}.self_attn.out_proj.bias` | Copy |
| `transformer.resblocks.{i}.ln_1.weight` | `text_model.encoder.layers.{i}.layer_norm1.weight` | Copy |
| `transformer.resblocks.{i}.ln_1.bias` | `text_model.encoder.layers.{i}.layer_norm1.bias` | Copy |
| `transformer.resblocks.{i}.ln_2.weight` | `text_model.encoder.layers.{i}.layer_norm2.weight` | Copy |
| `transformer.resblocks.{i}.ln_2.bias` | `text_model.encoder.layers.{i}.layer_norm2.bias` | Copy |
| `transformer.resblocks.{i}.mlp.c_fc.weight` | `text_model.encoder.layers.{i}.mlp.fc1.weight` | Copy (rename) |
| `transformer.resblocks.{i}.mlp.c_fc.bias` | `text_model.encoder.layers.{i}.mlp.fc1.bias` | Copy (rename) |
| `transformer.resblocks.{i}.mlp.c_proj.weight` | `text_model.encoder.layers.{i}.mlp.fc2.weight` | Copy (rename) |
| `transformer.resblocks.{i}.mlp.c_proj.bias` | `text_model.encoder.layers.{i}.mlp.fc2.bias` | Copy (rename) |
| `ln_final.weight` | `text_model.final_layer_norm.weight` | Copy |
| `ln_final.bias` | `text_model.final_layer_norm.bias` | Copy |
| `text_projection` | `text_projection.weight` | **Transpose** |

---

## 3. Phase 2 — Tokenizer Migration

> **File:** `custom_sd35.py` → `migrate_tokenizer()`

### What a tokenizer does

Before any text hits the neural network, it must be converted to integer token IDs. The tokenizer controls this mapping:

```
"a red ball" → [49406, 320, 736, 2721, 49407, 49407, ...]
 ↑ BOS token                                ↑ EOS + padding to 77 tokens
```

### Current implementation

```python
CLIC_LOCAL_DIR = "./CLIC-ViT-L-14-224-PixPr-RedCaps"

def migrate_tokenizer(openclip_model_name, save_dir):
    clic_dir = Path(CLIC_LOCAL_DIR)
    if clic_dir.exists():
        tokenizer = CLIPTokenizer.from_pretrained(str(clic_dir))
    else:
        tokenizer = CLIPTokenizer.from_pretrained("openai/clip-vit-large-patch14")
    
    tokenizer.model_max_length = 77
    tokenizer.save_pretrained(save_dir)
```

This loads the tokenizer **directly from the locally cloned CLIC model repository** (`./CLIC-ViT-L-14-224-PixPr-RedCaps/`), which contains the model's own `tokenizer.json`, `vocab.json`, and `merges.txt`. The code also verifies that the tokenizer's vocab size matches the model config's `vocab_size`.

For CLIC-ViT-L-14, the vocabulary is identical to standard CLIP (49408 BPE tokens), but loading from the model's own files is the correct practice — it guarantees the token IDs match the learned embeddings.

### The 77-token requirement

SD3.5 **always** pads/truncates to exactly 77 tokens. This is hardcoded in the pipeline. Setting `model_max_length = 77` ensures the tokenizer handles this correctly:

```python
tokens = tokenizer(
    "a red ball",
    padding="max_length",
    max_length=77,
    truncation=True,
    return_tensors="pt"
)
# tokens.input_ids.shape = [1, 77]  ← always
```

---

## 4. Phase 3 — Verification

> **File:** `custom_sd35.py` → `verify_conversion()`
> **File:** `run_full_workflow.py` → `verify_encoder_replacement()`

### Why verify

Weight surgery involves many index calculations and transposes. A single off-by-one error in the QKV split or a forgotten transpose will produce a model that *loads without errors* but produces *completely wrong embeddings*. Verification catches these bugs.

### Test A: Weight Drift Check (Static)

Compare raw weight tensors between OpenCLIP and converted HF model:

```python
# Token embeddings — should be bit-for-bit identical (just copied)
oc_token_emb = openclip_state['token_embedding.weight']
hf_token_emb = converted_state['text_model.embeddings.token_embedding.weight']
drift = (oc_token_emb - hf_token_emb).abs().max().item()
# drift should be 0.0 (exact copy)

# Q projection — derived from QKV split, should still be exact
oc_qkv = openclip_state['transformer.resblocks.5.attn.in_proj_weight']
oc_q = oc_qkv[:768, :]  # first third
hf_q = converted_state['text_model.encoder.layers.5.self_attn.q_proj.weight']
drift = (oc_q - hf_q).abs().max().item()
# drift should be 0.0 (slicing is exact)

# Text projection — transposed, so compare transposed version
oc_proj = openclip_state['text_projection'].T
hf_proj = converted_state['text_projection.weight']
drift = (oc_proj - hf_proj).abs().max().item()
# drift should be 0.0
```

**Pass criteria:** All drifts should be exactly `0.0` or at most `< 1e-7` (floating point noise from save/load).

### Test B: Forward Pass Comparison (Dynamic)

Run the **same text** through both models and compare outputs.

**What the current code does (functional validation only):**

```python
# Only checks that the HF model produces non-NaN output
tokens_hf = tokenizer_hf([text], return_tensors='pt', padding=True).to(DEVICE)
out_hf = converted_model(**tokens_hf)
assert not torch.isnan(out_hf.text_embeds).any()
```

**What it SHOULD also do (head-to-head comparison):**

```python
# OpenCLIP forward pass
tokenizer_oc = open_clip.get_tokenizer(OPENCLIP_CHECKPOINT)
tokens_oc = tokenizer_oc([text]).to(DEVICE)
with torch.no_grad():
    out_oc = openclip_model.encode_text(tokens_oc)
    out_oc = out_oc / out_oc.norm(dim=-1, keepdim=True)   # L2 normalize

# HF forward pass
tokens_hf = tokenizer_hf(text, return_tensors='pt', padding='max_length',
                          max_length=77, truncation=True).input_ids.to(DEVICE)
with torch.no_grad():
    out_hf = converted_model(tokens_hf)
    pooled_hf = out_hf.text_embeds                         # already normalized

# Compare
cosine_sim = torch.nn.functional.cosine_similarity(out_oc, pooled_hf)
l2_diff = (out_oc - pooled_hf).abs().max().item()
print(f"Cosine similarity: {cosine_sim.item():.6f}")  # should be > 0.999
print(f"Max L2 difference: {l2_diff:.2e}")             # should be < 1e-5
```

**⚠️ This head-to-head comparison is currently MISSING from the code.** See [Section 7](#7-known-issues--gaps-in-current-code).

### Test C: Pipeline-Level Replacement Verification

After the pipeline is assembled, `verify_encoder_replacement()` in `run_full_workflow.py` checks:

1. **Activation function:** `pipe.text_encoder.config.hidden_act == "quick_gelu"` (CLIC uses QuickGELU; if it were still the original, this would also be `quick_gelu` for SD3.5's CLIP-L, so this check alone is insufficient — see Issues)
2. **Weight comparison:** Pipeline's `text_encoder` weights must match `converted_text_encoder/`, not the original SD3.5 `text_encoder/`
3. **text_encoder_2 untouched:** Must NOT have `quick_gelu` (SD3.5's CLIP-G uses standard `gelu`)
4. **Forward pass:** The custom encoder produces valid output of correct shape `[1, 77, 768]`

---

## 5. Phase 4 — Component-Based Pipeline Integration

> **File:** `custom_sd35.py` → `create_sd35_with_custom_encoder()`

### Why load components individually

If you load the pipeline with `StableDiffusion3Pipeline.from_pretrained(...)`, it loads ALL components from the same directory. You can't easily swap just one encoder. Instead, we load each component separately and assemble them:

```python
# Load 9 components individually
transformer = SD3Transformer2DModel.from_pretrained(LOCAL_SD_DIR, subfolder="transformer", ...)
vae        = AutoencoderKL.from_pretrained(LOCAL_SD_DIR, subfolder="vae", ...)
scheduler  = FlowMatchEulerDiscreteScheduler.from_pretrained(LOCAL_SD_DIR, subfolder="scheduler")

text_encoder   = CLIPTextModelWithProjection.from_pretrained(CONVERTED_DIR, ...)   # ← CUSTOM
text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(LOCAL_SD_DIR, subfolder="text_encoder_2", ...)
text_encoder_3 = T5EncoderModel.from_pretrained(LOCAL_SD_DIR, subfolder="text_encoder_3", ...)

tokenizer   = CLIPTokenizer.from_pretrained(CONVERTED_DIR)          # ← CUSTOM
tokenizer_2 = CLIPTokenizer.from_pretrained(LOCAL_SD_DIR, subfolder="tokenizer_2")
tokenizer_3 = T5TokenizerFast.from_pretrained(LOCAL_SD_DIR, subfolder="tokenizer_3")
```

### Assembly

```python
pipe = StableDiffusion3Pipeline(
    transformer=transformer,
    vae=vae,
    scheduler=scheduler,
    text_encoder=text_encoder,       # ← OUR CUSTOM CLIC
    text_encoder_2=text_encoder_2,   # ← ORIGINAL CLIP-G
    text_encoder_3=text_encoder_3,   # ← ORIGINAL T5-XXL
    tokenizer=tokenizer,
    tokenizer_2=tokenizer_2,
    tokenizer_3=tokenizer_3,
)
```

### Why QuickGELU must be patched AFTER loading

When you call `from_pretrained()`, HuggingFace instantiates a fresh model from the config and loads weights. Even though the config says `hidden_act="quick_gelu"`, some `transformers` versions don't wire up the actual `QuickGELU` activation function. So we manually patch it:

```python
text_encoder = CLIPTextModelWithProjection.from_pretrained(CONVERTED_DIR, ...)
text_encoder = patch_quickgelu_activation(text_encoder)  # Force QuickGELU in all 12 layers
```

### How SD3.5 uses the three encoders internally

When you call `pipe("a red ball above a blue cube")`:

1. The prompt is tokenized **three times** (by `tokenizer`, `tokenizer_2`, `tokenizer_3`)
2. Each tokenized input goes through its respective encoder
3. Outputs are concatenated along the hidden dimension: `[768 + 1280 + 4096] = 6144`
4. The combined 6144-dim embedding conditions the MMDiT transformer at every denoising step
5. The pooled outputs (from CLIP-L and CLIP-G) are used as additional global conditioning

---

## 6. Phase 5 — Fine-Tuning Strategy

> **Status:** Documented but NOT yet implemented in code.

### Why fine-tuning is needed

The MMDiT backbone was trained to interpret embeddings from SD3.5's **original** CLIP-L encoder. Your custom CLIC encoder produces embeddings in a different space (different training data, different semantics). Even though the architecture is identical (768-dim output), the **meaning** of each dimension differs. The MMDiT needs to learn to "read" the new encoder's language.

### Recommended approach

```python
# Freeze the new text encoder (it's already trained, don't modify it)
for param in pipe.text_encoder.parameters():
    param.requires_grad = False

# Freeze the other two encoders too (they haven't changed)
for param in pipe.text_encoder_2.parameters():
    param.requires_grad = False
for param in pipe.text_encoder_3.parameters():
    param.requires_grad = False

# Only train the MMDiT backbone
for param in pipe.transformer.parameters():
    param.requires_grad = True

# Use a low learning rate since we're adapting, not training from scratch
optimizer = torch.optim.AdamW(pipe.transformer.parameters(), lr=1e-5)
```

### What to train on

Use a dataset of `(prompt, target_image)` pairs. The training loop:

1. Encode the image with VAE → latent representation
2. Add noise at a random timestep (flow matching)
3. Encode the prompt through all 3 text encoders → conditioning
4. Predict the noise with MMDiT (using conditioning)
5. Compute loss between predicted and actual noise
6. Backpropagate through MMDiT only (encoders are frozen)

### Alternative: No fine-tuning (zero-shot)

If both the original and custom CLIP-L were trained on similar data with similar objectives, the embedding spaces may be similar enough that the MMDiT can interpret them without fine-tuning. The test images (Phase 4) help evaluate this. If the images look coherent but slightly off, fine-tuning will help. If they look like random noise, something is wrong with the conversion.

---

## 7. Known Issues & Gaps in Current Code

### ✅ Issue 1 (FIXED): No head-to-head forward pass comparison

**Where:** `verify_conversion()` in `custom_sd35.py`

**Problem:** The verification only checks that the converted model produces non-NaN outputs with reasonable norms. It does **not** compare the output vectors of the OpenCLIP model vs. the converted HF model on the same input text. This means a subtle conversion bug (e.g., swapped K and V) could go undetected.

**Fix needed:** Add a comparison like this after the existing checks:

```python
# Run same text through OpenCLIP
tokenizer_oc = open_clip.get_tokenizer(OPENCLIP_CHECKPOINT)
tokens_oc = tokenizer_oc(test_texts).to(DEVICE)
with torch.no_grad():
    out_oc = openclip_model.encode_text(tokens_oc)
    out_oc = out_oc / out_oc.norm(dim=-1, keepdim=True)

# Run same text through converted HF model
tokens_hf = tokenizer_hf(
    test_texts, return_tensors='pt', padding='max_length',
    max_length=77, truncation=True
).input_ids.to(DEVICE)
with torch.no_grad():
    out_hf = converted_model(tokens_hf).text_embeds

# They should be nearly identical
diff = (out_oc - out_hf).abs().max().item()
assert diff < 1e-4, f"Forward pass mismatch: {diff}"
```

### ✅ Issue 2 (FIXED): `bos_token_id` / `eos_token_id` config mismatch

**Where:** `custom_sd35.py` → `convert_openclip_to_transformers()`

**Fix applied:** The `CLIPTextConfig` now explicitly sets `bos_token_id=0` and `eos_token_id=2` to match SD3.5's original `text_encoder/config.json`. Re-run conversion to regenerate the config.

### ✅ Issue 3 (FIXED): Activation check is not definitive

**Where:** `verify_encoder_replacement()` in `run_full_workflow.py`

**Fix applied:** The comment and log message now clarify that both the original and custom CLIP-L use `quick_gelu`, so this check alone is not proof of replacement. The weight comparison in step [3/5] is correctly identified as the definitive verification.

### ⚠️ Issue 4: Fine-tuning (Phase 5) not implemented

The code generates images with the swapped encoder but doesn't include any training loop. The MMDiT will interpret the new encoder's embeddings using its original training, which may produce degraded images. Implementing the fine-tuning loop described in Section 6 would likely improve results.

### ✅ Issue 5 (FIXED): Tokenizer migration doesn't verify vocabulary match

**Where:** `migrate_tokenizer()` in `custom_sd35.py`

**Fix applied:** The function now loads the tokenizer from the locally cloned CLIC model directory (`./CLIC-ViT-L-14-224-PixPr-RedCaps/`) and asserts that `len(tokenizer) == config.vocab_size` before saving.

---

## 8. Quick Reference: File Map

| File | Purpose |
|---|---|
| `custom_sd35.py` | Core library: conversion, tokenizer, verification, pipeline assembly |
| `run_full_workflow.py` | End-to-end script: runs Phase 1→4 sequentially, generates test images |
| `verify_weight_changes.py` | Standalone weight comparison (converted vs. original SD3.5) |
| `open_to_hf.py` | Vision encoder conversion (separate, for the vision side) |
| `full_custom_sd35_job.sh` | SLURM job script to run the full workflow on cluster |
| `converted_text_encoder/` | Output directory with converted model + tokenizer |
| `stable-diffusion-3.5-large-turbo/` | Original SD3.5 model directory |
| `test_outputs/` | Basic generation test images |
| `spatial_outputs/` | Spatial reasoning test images |

### Running the full pipeline

```bash
# On SLURM cluster:
sbatch full_custom_sd35_job.sh

# Or directly:
conda activate sd2
python run_full_workflow.py
```

### Expected output structure after running

```
converted_text_encoder/
├── config.json              # HF config with quick_gelu
├── model.safetensors        # Converted weights
├── merges.txt               # BPE merges (from CLIP tokenizer)
├── vocab.json               # BPE vocabulary
├── tokenizer_config.json    # Tokenizer settings
└── special_tokens_map.json  # Special token definitions

test_outputs/
├── test_01.png  ... test_04.png

spatial_outputs/
├── spatial_01_*.png ... spatial_15_*.png
```

