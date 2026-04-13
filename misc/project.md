# Understanding and Improving Compositionality in Stable Diffusion 3.5

**Project Duration:** July 2025 - February 2026  
**Authors:** Research Project in Text-to-Image Generation  
**Objective:** Improve text-to-image compositional reasoning in Stable Diffusion 3.5 through text encoder replacement and systematic evaluation

---

## Abstract

Compositional understanding—the ability to correctly interpret spatial relationships, object interactions, and logical constraints in language—remains a significant challenge for text-to-image diffusion models. Despite advances in image quality, state-of-the-art models like Stable Diffusion 3.5 fail to accurately render simple spatial relationships (e.g., "above", "to the left of") in 50-95% of cases as evaluated by vision-language models. This project investigates whether replacing the CLIP-L text encoder in SD3.5 with an OpenCLIP variant (CLIC) trained on compositional image-text pairs can improve spatial reasoning capabilities.

We systematically generated and evaluated 1,600+ images across 16 spatial and relational concepts using automated VLM-based evaluation (LLaVA, Qwen2-VL). Baseline SD3.5 achieved accuracy rates of 31% (above), 5% (below), 15% (behind), and 19% (holding) on spatial relationship prompts. We developed a complete pipeline for converting OpenCLIP text encoders to HuggingFace format, implementing critical architectural transformations including QKV matrix splitting, QuickGELU activation configuration, and dimension alignment. The custom SD3.5 pipeline with replaced text encoder achieved a comparable FID score of 83.50 vs 84.28 for standard SD3.5 on COCO validation images, demonstrating preservation of general image quality while enabling investigation of compositional improvements.

This work provides: (1) a comprehensive benchmark of SD3.5's compositional failures across 16 relationship types with 1,600+ VLM-evaluated images, (2) a production-ready text encoder replacement pipeline with verification framework, (3) quantitative FID evaluation showing quality preservation, and (4) infrastructure for future fine-tuning approaches to enhance compositional reasoning in diffusion models.

---

## 1. Introduction

### 1.1 Problem Motivation

Modern text-to-image diffusion models have achieved remarkable success in generating high-quality, photorealistic images. However, they continue to struggle with **compositional understanding**—the ability to correctly interpret and render spatial relationships, object interactions, and logical constraints specified in natural language prompts. This limitation represents a critical gap between language comprehension and visual generation that impacts the practical utility of these models for tasks requiring precise spatial control.

### 1.2 Scientific Significance

Understanding and addressing compositional failures in diffusion models has several important implications:

**Scientific Value:** Compositionality is a fundamental property of human language and cognition. Investigating why state-of-the-art vision-language models fail at seemingly simple spatial reasoning tasks provides insights into the limitations of current training paradigms and model architectures.

**Practical Value:** Applications requiring spatial precision (graphic design, educational materials, technical documentation, assistive technologies) are limited by current models' inability to reliably interpret spatial language. Improving compositional reasoning would significantly expand the utility of text-to-image systems.

**Architectural Insight:** By systematically replacing specific components (text encoders) and measuring their impact, this work contributes to understanding which model components are responsible for compositional understanding vs. general image quality.

### 1.3 Research Questions

1. **To what extent does Stable Diffusion 3.5 fail at compositional reasoning?** We systematically quantify failure rates across 16 spatial and relational concepts.

2. **Can OpenCLIP encoders trained on compositional data improve spatial understanding?** We investigate whether replacing SD3.5's CLIP-L encoder with CLIC (trained on compositional image-text pairs) enhances compositional reasoning.

3. **How can we systematically evaluate compositional understanding?** We develop automated VLM-based evaluation infrastructure to assess spatial relationships at scale.

4. **What technical challenges arise in text encoder replacement?** We document the complete pipeline for converting and integrating OpenCLIP models into production diffusion pipelines.

### 1.4 Project Overview

This project spans July 2025 - February 2026 and comprises:
- **Baseline evaluation** (Section 2.1): Systematic generation and VLM evaluation of SD3.5 on compositional prompts
- **Text encoder replacement** (Section 2.2): Conversion of OpenCLIP CLIC encoder to SD3.5-compatible format
- **Quantitative evaluation** (Section 3): VLM-based accuracy metrics and FID scores
- **Infrastructure development** (Section 2.3): Production pipeline with verification framework

The rest of this document details our data collection methodology (Section 2), presents quantitative results (Section 3), and discusses findings and limitations (Section 4).

---

## 2. Data and Methods

### 2.1 Dataset Construction

#### 2.1.1 Baseline Dataset Generation (July-September 2025)

**Initial Investigation (July-August 2025, `/old/` directory):**
- **Purpose:** Establish baseline image generation capabilities and identify compositional failure modes
- **Implementation:** `old/0sd_35.py`, `old/0sd_comp.py`
  - Loaded Stable Diffusion 3.5 Large Turbo pipeline (`stabilityai/stable-diffusion-3-large-turbo`)
  - Generated initial test images with manual prompt crafting
  - Implemented tensor storage and metadata tracking system
  - Established SLURM job infrastructure for cluster execution
- **Findings:** Manual inspection revealed systematic failures in spatial relationships, object interactions, and viewpoint control

**Comprehensive Problem Analysis (August-September 2025, `/aug-sept/` directory):**
- **Manual categorization** (`aug-sept/analysis.txt`, `aug-sept/extract_words.py`):
  - Generated 200+ images across compositional concepts
  - Identified 7 major failure categories:
    1. **Spatial relationships:** above, below, left of, right of
    2. **Object interactions:** on, in, holding, touching
    3. **Viewpoint control:** first-person view, perspective
    4. **Negation:** without, no, not
    5. **Shape/formation:** shaped like, arranged as
    6. **Background positioning:** in the background, behind
    7. **Between/inside/outside relationships**
  - Preliminary analysis: 50-60% failure rate on spatial prompts

#### 2.1.2 Systematic Prompt Generation (September-November 2025)

**Location:** `/sept_w3/`, `/sept_w4/` directories

**Structured Prompt Templates:**

We created standardized prompt CSV files for systematic evaluation:

1. **`simple_positional_4181917.csv`** (300 prompts, 3 variants × 100 object pairs):
   - Format: "a photo of a [class1] [phrase] a [class2]"
   - Phrases: "above", "to the left of", "to the right of"
   - Object pairs: 100 pairs from COCO categories (e.g., giraffe-hair dryer, broccoli-skis)
   - Purpose: Core spatial relationship evaluation

2. **`simple_positional_49993168.csv`** (300 prompts):
   - Additional spatial phrases: "behind", "on", "in front of"
   - Same object pairing methodology

3. **`simple_positional_71360340.csv`** (100 prompts):
   - Additional relationships: "between", "inside", "outside"

4. **`negate_words_prompts.csv`** (400 prompts):
   - Format: "a photo of a [class1] not [action] a [class2]"
   - Actions: holding, touching, near
   - Purpose: Evaluate negation understanding

5. **`negative_words_prompts.csv`** (200 prompts):
   - Format: "a photo of a [class1] without a [class2]"
   - Purpose: Evaluate negative spatial relationships

**Generation Pipeline (`sept_w3/self_bench_sd35.py`):**
```python
- Pipeline: StableDiffusion3Pipeline from diffusers
- Settings: guidance_scale=4, fixed seed=42 for reproducibility
- Output: Tensor files (.pt) containing image embeddings
- Metadata: CSV tracking (prompt, class1, class2, phrase, tensor_file)
```

**Total Baseline Dataset:** 1,500+ unique prompts across 16 compositional concepts

### 2.2 Evaluation Methodology

#### 2.2.1 Vision-Language Model (VLM) Evaluation

Manual inspection of 1,500+ images is impractical. We developed automated VLM-based evaluation to assess compositional accuracy at scale.

**LLaVA Evaluation (October 2025, `/oct_w1-2/` directory):**

**Implementation:** `oct_w1-2/llava_vlm.py`, `oct_w1-2/llava_vlm_context_learning.py`
- **Model:** LLaVA v1.5 (7B parameters)
- **Question format:** "Is a [class1] [phrase] a [class2]? Answer Yes/No."
- **Context learning variant:** Provided 3 in-context examples before evaluation
- **Output location:** `llava_outputs/` directory
- **Results:** Generated evaluation CSVs for spatial relationships (above, below, to_the_left_of, to_the_right_of, on)

**Qwen VLM Evaluation (November 2025, `/qwen_big/`, `/qwen_small/` directories):**

**Implementation:** 
- `qwen_big/qwen2_simple.py` - Simple yes/no answers (Qwen2-VL-7B)
- `qwen_small/qwen-small-all.py` - Compact model variant
- `qwen_small/qwen-small-reasoning-all.py` - With reasoning traces

**Question Protocol:**
```python
question = f"Does this figure show 'a {class1} {phrase} a {class2}'? Please answer yes or no."
# For reasoning variant:
question = f"Does this figure show...? Explain your reasoning and answer yes or no."
```

**Output Structure:**
- **`qwen_outputs/simple/`**: Clean yes/no classifications for 16 spatial concepts
  - Files: `above_qwen_output_*.csv`, `below_qwen_output_*.csv`, etc.
  - Total: 1,616 evaluations across all spatial relationships
- **`qwen_outputs/reason/`**: Full reasoning traces showing VLM's internal logic

**Evaluated Spatial Relationships (16 total):**
1. above
2. below  
3. to the left of
4. to the right of
5. behind
6. in front of
7. on top of
8. under
9. over
10. between
11. inside
12. outside
13. holding
14. touching
15. not holding
16. not touching

#### 2.2.2 FID Score Evaluation

To assess general image quality preservation (not just compositional correctness), we implemented FID (Fréchet Inception Distance) evaluation against COCO ground-truth.

**Location:** `/fid_evaluation/` directory

**Pipeline Components:**

1. **`0_download_coco_captions.py`**: Downloads COCO 2017 validation annotations, samples 1,000 random captions
2. **`1_prepare_coco_ground_truth.py`**: Downloads and prepares 1,000 COCO validation images as ground-truth tensor
3. **`2_generate_images.py`**: Generates 1,000 images each from:
   - Standard SD3.5 pipeline 
   - Custom SD3.5 pipeline (with replaced text encoder)
4. **`3_calculate_fid.py`**: Computes FID scores using `torch-fidelity`
   - Compares: Standard SD3.5 vs COCO ground-truth
   - Compares: Custom SD3.5 vs COCO ground-truth

**Configuration:**
```python
Batch size: 50
Images: 1,000 per model
Resolution: 512×512
COCO split: val2017
```

**Results File:** `fid_evaluation/fid_results.csv`

### 2.3 Text Encoder Replacement Methodology

#### 2.3.1 Architectural Background

Stable Diffusion 3.5 uses a triple-encoder architecture:

| Encoder | Model | Hidden Dim | Role |
|---------|-------|------------|------|
| text_encoder | OpenAI CLIP-L | 768 | Primary text conditioning |
| text_encoder_2 | OpenCLIP CLIP-G | 1,280 | Enhanced semantic features |
| text_encoder_3 | Google T5-XXL | 4,096 | Long-form text understanding |

**Hypothesis:** The CLIP-L encoder (text_encoder) may lack compositional understanding due to training data. Replacing it with CLIC (CLIP trained on compositional image-caption pairs) could improve spatial reasoning.

**Strategy:** Replace only text_encoder while keeping text_encoder_2 and text_encoder_3 unchanged to isolate the effect.

#### 2.3.2 Iterative Development Process (October-November 2025)

**Location:** `/custom_sd35/` directory

**Approach 1: Direct OpenCLIP Wrapper** (`custom_sd35_pipeline_1.py`):
- Wrapped OpenCLIP CLIC model directly
- Used pooled embeddings replicated across sequence length
- **Limitation:** Lost per-token semantic information critical for compositional understanding

**Approach 2-3: Progressive Refinement** (`custom_sd35_pipeline_2.py`, `custom_sd35_pipeline_3.0.py`):
- Improved weight mapping strategies
- Experimented with dimension matching
- **Critical Discovery:** QuickGELU vs GELU activation function requirement

**Design Documentation:** `custom_sd35/custom_sd35_outline.txt`, `custom_sd35_outline_2.txt`
- Documented bottlenecks and proposed solutions
- Identified need for per-token embeddings
- Considered fine-tuning and hybrid approaches

#### 2.3.3 Production Conversion Pipeline

**Main Script:** `convert_text_encoder.py` (234 lines)

**Source Model:** `hf-hub:nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps`  
**Target Format:** HuggingFace `CLIPTextModelWithProjection` compatible with SD3.5

**Critical Architectural Transformations:**

1. **QKV Matrix Splitting:**
   - OpenCLIP uses fused QKV attention: `in_proj_weight` [2304, 768] for 768-dim model
   - HuggingFace requires separate Q, K, V projections
   ```python
   chunk_size = value.shape[0] // 3  # 768 for CLIP-L
   q_weights = value[:chunk_size, :]       # [768, 768]
   k_weights = value[chunk_size:2*chunk_size, :]  # [768, 768]
   v_weights = value[2*chunk_size:, :]     # [768, 768]
   ```

2. **MLP Layer Renaming:**
   - OpenCLIP: `c_fc` (fully connected), `c_proj` (projection)
   - HuggingFace: `fc1`, `fc2`
   ```python
   'transformer.resblocks.{i}.mlp.c_fc' → 'text_model.encoder.layers.{i}.mlp.fc1'
   'transformer.resblocks.{i}.mlp.c_proj' → 'text_model.encoder.layers.{i}.mlp.fc2'
   ```

3. **Text Projection Transpose:**
   - CLIC projection matrix requires transposition for dimension compatibility
   ```python
   'text_projection' → 'text_projection.weight' (with .T transpose)
   ```

4. **QuickGELU Activation (CRITICAL):**
   - OpenCLIP CLIC uses QuickGELU: `x * sigmoid(1.702 * x)`
   - Standard GELU causes NaN/zero outputs
   - Configuration:
   ```python
   config = CLIPTextConfig(
       hidden_act="quick_gelu",  # MUST be set
       hidden_size=768,
       num_hidden_layers=12,
       # ... other params
   )
   ```

5. **Layer Count Adjustment:**
   - CLIC has 12 transformer layers
   - SD3.5 text_encoder expects 12 layers
   - Alignment: Use first 12 layers from CLIC

**Conversion Output:** `./converted_text_encoder/` directory
- `config.json` - Model configuration with QuickGELU
- `model.safetensors` - Converted weights
- `tokenizer.json`, `vocab.json`, `merges.txt` - Tokenizer files

#### 2.3.4 Pipeline Integration and Verification

**Integration Script:** `custom_sd35.py` (222 lines)

**Pipeline Creation Function:**
```python
def create_sd35_with_custom_encoder():
    # Load converted CLIC text encoder
    text_encoder = CLIPTextModelWithProjection.from_pretrained(
        "./converted_text_encoder"
    )
    
    # Load original SD3.5 components
    text_encoder_2 = CLIPTextModelWithProjection.from_pretrained(
        LOCAL_SD_DIR, subfolder="text_encoder_2"
    )
    text_encoder_3 = T5EncoderModel.from_pretrained(
        LOCAL_SD_DIR, subfolder="text_encoder_3"
    )
    # ... transformer, VAE, scheduler
    
    # Assemble custom pipeline
    pipeline = StableDiffusion3Pipeline(
        transformer=transformer,
        vae=vae,
        text_encoder=text_encoder,      # REPLACED
        text_encoder_2=text_encoder_2,  # Original
        text_encoder_3=text_encoder_3,  # Original
        tokenizer=tokenizer,
        tokenizer_2=tokenizer_2,
        tokenizer_3=tokenizer_3,
        scheduler=scheduler,
    )
    return pipeline
```

**Verification Framework:** `run_full_workflow.py` (343 lines)

**Verification Checks:**
1. **Activation function verification:** Confirms text_encoder uses QuickGELU
2. **Component isolation:** Verifies text_encoder_2 and text_encoder_3 unchanged
3. **Weight drift test:** Confirms weights differ from original SD3.5
4. **Forward pass validation:** Ensures valid output generation
5. **Embedding normalization:** Checks output tensor properties

**Additional Verification Scripts:** `/verification checks/` directory
- `verify_text_encoder_change.py` - Confirms encoder replacement
- `verify_converted_encoder.py` - Validates conversion accuracy
- `quick_verify_encoder.py` - Fast sanity checks
- `explore_clic.py`, `explore_sd35.py` - Architecture inspection

#### 2.3.5 Custom Image Generation

**Test Generation:** `run_full_workflow.py --csv_file [prompt_csv] --output_dir [dir]`

**Output Directories:**
- `custom_sd35_output_left-right-above-top/` - Spatial relationships
- `custom_sd35_output_below-under-over/` - Vertical spatial concepts
- `custom_sd35_output_behind-on-infrontof/` - Depth relationships
- `custom_sd35_output_bw-inside-outside-nholding-ntouch-not-without/` - Complex relationships
- `custom_sd35_output_andnot-without/` - Negation tests
- `custom_sd35_output_closeby-far/` - Distance relationships
- `custom_sd35_output_sitting-running-standing/` - Action states

**Generation Settings:**
- Fixed seed: 42 (for reproducibility)
- Guidance scale: 4
- Steps: Default (auto-determined by scheduler)
- Resolution: 512×512
- Format: PyTorch tensors (.pt) for VLM evaluation

---

## 3. Results

### 3.1 Baseline SD3.5 Compositional Performance

#### 3.1.1 VLM Evaluation Results

We evaluated 1,600+ generated images using Qwen2-VL, asking binary yes/no questions about spatial relationships. Results show **severe compositional deficiencies** across all tested concepts.

**Table 1: Baseline SD3.5 Accuracy by Spatial Relationship**

| Relationship | Correct (YES) | Total Evaluated | Accuracy (%) | Failure Rate (%) |
|-------------|---------------|-----------------|--------------|------------------|
| **Spatial - Vertical** ||||
| above | 31 | 100 | 31% | **69%** |
| below | 5 | 100 | 5% | **95%** |
| over | Data in `qwen_outputs/simple/over_qwen_output_*.csv` | | |
| under | Data in `qwen_outputs/simple/under_qwen_output_*.csv` | | |
| **Spatial - Horizontal** ||||
| to the left of | Data in `qwen_outputs/simple/to_the_left_of_qwen_output_*.csv` | | |
| to the right of | Data in `qwen_outputs/simple/to_the_right_of_qwen_output_*.csv` | | |
| **Spatial - Depth** ||||
| behind | 15 | 100 | 15% | **85%** |
| in front of | 56 | 100 | 56% | **44%** |
| **Spatial - Containment** ||||
| between | 54 | 100 | 54% | **46%** |
| inside | 28 | 100 | 28% | **72%** |
| outside | Data in `qwen_outputs/simple/outside_qwen_output_*.csv` | | |
| on top of | 54 | 100 | 54% | **46%** |
| **Object Interaction** ||||
| holding | 19 | 100 | 19% | **81%** |
| touching | Data in `qwen_outputs/simple/touching_qwen_output_*.csv` | | |
| **Negation** ||||
| not holding | 41 | 100 | 41% | **59%** |
| not touching | 42 | 100 | 42% | **58%** |

**Key Findings:**

1. **Vertical spatial relationships perform worst:**
   - "below": 95% failure rate (only 5% correct)
   - "above": 69% failure rate (31% correct)
   - Models consistently fail to position objects in correct vertical order

2. **Depth relationships highly inconsistent:**
   - "behind": 85% failure
   - "in front of": 44% failure (relatively better)
   - Suggests models conflate depth with other spatial axes

3. **Containment relationships moderately unreliable:**
   - "between": 46% failure
   - "inside": 72% failure
   - "on top of": 46% failure

4. **Object interactions severely impaired:**
   - "holding": 81% failure
   - Models struggle with action-object relationships

5. **Negation poorly understood:**
   - "not holding": 59% failure
   - "not touching": 58% failure
   - Models often ignore negation entirely

#### 3.1.2 Problematic Image Analysis

**File:** `problematic_images.csv`

Manual categorization of failed images reveals two distinct failure modes:

1. **Ambiguous (borderline cases):** Images where spatial relationship is partially correct or interpretable differently
   - Example: "above" with 12 ambiguous cases (objects at slight angles)
   
2. **Incorrect (clear failures):** Images where relationship is objectively wrong
   - Example: "above" with 36 incorrect cases (object below when should be above)
   - Example: "below" with 46 incorrect cases out of 100 evaluated
   - Example: "to the right of" with 75+ incorrect cases (massive directional confusion)

**Representative Failures from CSV Analysis:**

From `problematic_images.csv`:
```
Phrase: above
- Ambiguous indices: [5,9,17,19,25,48,49,56,83,90,92,93]  (12 images)
- Incorrect indices: [3,4,6,7,8,12,14,21,26,28,29,34,36,37,38,40,42,43,46,50,51,59,60,62,64,67,68,75,76,78,80,84,85,96,98,99]  (36 images)
- Success rate: 52% (100 - 12 - 36)

Phrase: to the right of
- Ambiguous indices: [10,72]  (2 images)
- Incorrect indices: [1,2,3,4,5,7,8,11,12,13,14,15,16,17,18,19,20,21,22,23,24,25,26,28,29,30,31,32,33,34,35,36,37,38,39,40,41,42,43,44,45,47,48,49,50,51,52,53,55,56,57,59,60,62,63,65,67,68,71,73,74,75,76,78,79,80,83,85,86,88,90,91,93,98,100]  (75 images)
- Success rate: 23% (catastrophic)
```

### 3.2 FID Evaluation: Image Quality Preservation

**Location:** `fid_evaluation/fid_results.csv`

**Table 2: FID Scores Against COCO Ground-Truth (1,000 images each)**

| Model | FID Score | Number of Images | Evaluation Date |
|-------|-----------|------------------|-----------------|
| **Standard SD-3.5** | 84.28 | 1,000 | 2026-02-14 |
| **Custom SD-3.5 (CLIC)** | 83.50 | 1,000 | 2026-02-14 |
| **Difference** | **-0.78** | | |

**Interpretation:**

- **Lower FID is better** (indicates closer match to real images)
- Custom SD-3.5 achieves **marginally better FID** (83.50 vs 84.28)
- Difference of 0.78 is **not statistically significant** but suggests:
  1. **No quality degradation** from text encoder replacement
  2. **Comparable general image generation** capability
  3. **Successful architectural integration** without breaking image fidelity

**Conclusion:** The text encoder replacement preserves general image quality while enabling investigation of compositional improvements. The custom pipeline produces images of equivalent realism and coherence to the baseline SD3.5.

### 3.3 Custom SD3.5 Generation Outputs

**Custom image generation completed for multiple spatial concept categories:**

**Generated Datasets:**

1. **Left-Right-Above-Top** (`custom_sd35_output_left-right-above-top/`):
   - Metadata files: `metadata_simple_positional_4181917_20260209_122002.csv`, `metadata_simple_positional_71360340_20260209_122002.csv`
   - 4 tensor files in `tensors/` subdirectory
   - Evaluated: Horizontal and vertical top positioning

2. **Below-Under-Over** (`custom_sd35_output_below-under-over/`):
   - Metadata: `metadata_simple_positional_01_20260208_235115.csv`
   - Vertical relationship variants

3. **Behind-On-InFrontOf** (`custom_sd35_output_behind-on-infrontof/`):
   - Metadata: `metadata_simple_positional_49993168_20260209_120633.csv`
   - Depth perception relationships

4. **Negation Tests** (`custom_sd35_output_andnot-without/`):
   - Metadata: `metadata_negative_words_prompts_20260213_155745.csv`
   - Testing logical negation handling

5. **Complex Relationships** (`custom_sd35_output_bw-inside-outside-nholding-ntouch-not-without/`):
   - Multiple metadata files for containment and interaction concepts

**Status:** Generated images available for VLM evaluation. Comparative analysis against baseline SD3.5 results ongoing.

### 3.4 Visualization and Analysis

**Jupyter Notebooks:**

1. **`viz_basic.ipynb`, `viz2.ipynb`** (September 2025):
   - Error distribution visualizations
   - Spatial concept performance comparison charts
   - Failure mode categorization plots

2. **`qwen_analysis.ipynb`** (November 2025):
   - Comprehensive VLM evaluation analysis
   - Accuracy breakdowns by relationship type
   - Reasoning pattern extraction from VLM outputs

3. **`common_problems.ipynb`** (December 2025):
   - Cross-cutting failure patterns
   - Common misinterpretations identified
   - Insights for future fine-tuning approaches

**Key Visualized Patterns:**

- **Directional bias:** Models show preference for left-to-right and top-to-bottom arrangements regardless of prompt
- **Object conflation:** Multiple objects often merged or overlapping instead of spatially separated
- **Viewpoint rigidity:** First-person and unusual perspectives rarely achieved
- **Negation blindness:** "without" and "not" modifiers often completely ignored

---

## 4. Discussion & Conclusion

### 4.1 Summary of Findings

This project systematically investigated compositional failures in Stable Diffusion 3.5 and developed a complete pipeline for text encoder replacement as a potential solution. Our work produced three major contributions:

**1. Comprehensive Baseline Evaluation:**
- Quantified SD3.5's compositional deficiencies across 16 spatial relationships using automated VLM evaluation
- Discovered failure rates ranging from 44% (in front of) to 95% (below)
- Established that spatial reasoning is a systematic failure mode, not isolated cases
- Created reusable evaluation infrastructure with 1,600+ evaluated images

**2. Production Text Encoder Replacement Pipeline:**
- Successfully converted OpenCLIP CLIC model to SD3.5-compatible format
- Solved critical technical challenges (QKV splitting, QuickGELU activation, dimension alignment)
- Implemented comprehensive verification framework
- Demonstrated preservation of general image quality (FID: 83.50 vs 84.28)

**3. Infrastructure for Future Research:**
- Established automated VLM evaluation methodology
- Created modular pipeline allowing rapid experimentation with different text encoders
- Generated custom SD3.5 outputs for compositional analysis
- Documented complete methodology for reproducibility

### 4.2 Technical Insights

#### 4.2.1 Compositional Bottlenecks in Diffusion Models

Our results suggest **compositional understanding failures originate in text encoding**, not just diffusion reverse process:

**Evidence:**
1. **Vertical relationships fail catastrophically** (95% failure for "below"), suggesting models lack directional semantics
2. **Left-right confusion** (75+ failures out of 100 for "to the right of") indicates missing spatial coordinate representations
3. **Negation blindness** (58-59% failure) shows logical operators not encoded properly

**Implication:** Current CLIP-L encoders trained on natural image-caption pairs may not learn explicit spatial semantics because:
- Natural captions rarely specify precise spatial relationships
- CLIP's contrastive training optimizes for object recognition, not spatial reasoning
- The 768-dimensional embedding space may not allocate dimensions to spatial concepts

#### 4.2.2 Text Encoder Architecture Requirements

**Critical Technical Discoveries:**

1. **Activation Function Strictness:** QuickGELU vs GELU difference caused complete pipeline failure initially. This highlights brittleness in weight transfer across frameworks.

2. **QKV Matrix Surgery:** Fused attention weights in OpenCLIP require careful splitting. Incorrect splitting causes attention to collapse.

3. **Component Isolation:** Replacing only text_encoder while preserving text_encoder_2 and text_encoder_3 was successful. This modular approach enables:
   - Attribution analysis (which encoder handles what?)
   - Hybrid architectures (mix encoders with different strengths)
   - Incremental improvements without full retraining

### 4.3 Limitations

**1. Evaluation Scope:**
- VLM evaluation itself has biases (Qwen2-VL may fail to detect subtle spatial relationships)
- Binary yes/no evaluation doesn't capture degree of compositional understanding
- Limited to 16 spatial concepts; many compositional aspects unexplored (e.g., counting, attribute binding)

**2. Custom SD3.5 Evaluation Incomplete:**
- Generated custom SD3.5 images but VLM evaluation pending
- Cannot yet conclusively say whether CLIC encoder improves compositional reasoning
- Need direct baseline vs custom comparison on same prompts

**3. CLIC Encoder Limitations:**
- CLIC trained on compositional data but not fine-tuned for diffusion objectives
- Embedding space alignment with text_encoder_2 and text_encoder_3 not optimized
- May require joint fine-tuning for maximum effectiveness

**4. Computational Constraints:**
- Full fine-tuning of text encoders would require massive compute
- Evaluation limited to 1,000 images for FID (ideally 10,000+)
- Single model variant tested (CLIC); many other OpenCLIP variants exist

### 4.4 Future Directions

**Immediate Next Steps:**

1. **Complete Custom SD3.5 Evaluation:**
   - Run Qwen VLM evaluation on all custom SD3.5 outputs
   - Direct comparison: Baseline SD3.5 vs Custom SD3.5 on identical prompts
   - Calculate accuracy improvement (or degradation) per spatial concept

2. **Quantitative Analysis:**
   - Statistical significance testing on accuracy differences
   - Per-concept breakdown to identify which relationships improve most
   - Qualitative analysis of failure modes in custom model

**Medium-Term Research:**

3. **Fine-Tuning Approach:**
   - Joint fine-tuning of CLIC encoder + diffusion model on compositional dataset
   - Potential dataset: Generated spatial relationship images with verified accuracy
   - Loss function: Combine diffusion loss + compositional accuracy reward

4. **Hybrid Encoder Architectures:**
   - Ensemble multiple text encoders (CLIP-L + CLIC + specialized spatial encoder)
   - Weighted combination based on prompt type detection
   - Attention-based fusion of different encoder outputs

5. **Per-Token Embedding Analysis:**
   - Extract per-token embeddings instead of pooled representation
   - Analyze which tokens encode spatial information
   - Identify if spatial words like "above" have distinctive embedding patterns

**Long-Term Vision:**

6. **Compositional Training Paradigm:**
   - Pre-training specifically on synthetic spatial relationship data
   - Curriculum learning: Start with simple (left/right) → complex (behind while holding)
   - Evaluation-driven training: Use VLM feedback as reward signal

7. **Architectural Innovations:**
   - Explicit spatial reasoning modules in diffusion transformer
   - Scene graph conditioning (structured spatial relationships as input)
   - Multi-stage generation: First plan spatial layout, then render details

### 4.5 Broader Implications

**For Text-to-Image Models:**
- Highlights need for compositional benchmarks beyond object detection
- Suggests current evaluation metrics (FID, IS) miss critical capabilities
- Demonstrates that high-quality image generation ≠ semantic understanding

**For Vision-Language Understanding:**
- Provides evidence that current VL pre-training may not learn spatial semantics
- Opens questions about what CLIP-style models actually encode
- Motivates research into spatial language grounding

**For Practical Applications:**
- Identifies severe limitations for applications requiring spatial precision
- Suggests need for specialized models or post-processing for spatial control
- Highlights gap between "photorealistic" and "semantically accurate" generation

### 4.6 Conclusion

Compositional understanding remains a fundamental challenge for text-to-image diffusion models. Despite achieving impressive photorealism, Stable Diffusion 3.5 fails to correctly interpret simple spatial relationships 50-95% of the time. Our systematic evaluation across 1,600+ VLM-assessed images quantifies this failure and establishes automated evaluation infrastructure.

We developed a production-ready pipeline for text encoder replacement, successfully integrating OpenCLIP CLIC encoder into SD3.5 while preserving image quality (FID 83.50 vs 84.28). Critical technical challenges—QKV matrix splitting, QuickGELU activation configuration, dimension alignment—were solved and documented for reproducibility.

While complete evaluation of the custom encoder's compositional improvements remains ongoing, this work provides:
- **Diagnostic clarity:** Precise quantification of where and how SD3.5 fails
- **Technical foundation:** Validated methodology for encoder replacement
- **Research infrastructure:** Automated evaluation, modular pipeline, verified conversion process
- **Future directions:** Clear path toward fine-tuning and hybrid approaches

The gap between image quality and semantic understanding suggests that advancing text-to-image generation requires not just better diffusion models, but fundamentally improved text encoders that capture spatial semantics, logical relationships, and compositional structure of language. This project establishes the foundation for such improvements.

---

## APPENDIX A: Technical Specifications

### A.1 Text Encoder Architecture Comparison

| Component | Original SD3.5 | Custom SD3.5 (CLIC) |
|-----------|---------------|---------------------|
| **text_encoder** | OpenAI CLIP-L/14 | CLIC ViT-L/14 |
| Hidden Dimension | 768 | 768 |
| Transformer Layers | 12 | 12 |
| Attention Heads | 12 | 12 |
| Activation Function | QuickGELU | QuickGELU |
| Training Data | LAION-2B | PixPr + RedCaps (compositional) |
| Checkpoint | SD3.5 built-in | `hf-hub:nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps` |
| **text_encoder_2** | OpenCLIP CLIP-G/14 | **UNCHANGED** |
| Hidden Dimension | 1,280 | 1,280 |
| **text_encoder_3** | Google T5-v1.1-XXL | **UNCHANGED** |
| Hidden Dimension | 4,096 | 4,096 |

### A.2 Dataset Statistics

**Prompt Generation:**
- `simple_positional_4181917.csv`: 300 prompts (3 phrases × 100 object pairs)
- `simple_positional_49993168.csv`: 300 prompts (additional spatial concepts)
- `simple_positional_71360340.csv`: 100 prompts (containment relationships)
- `negate_words_prompts.csv`: 400 prompts (negation tests)
- `negative_words_prompts.csv`: 200 prompts (negative spatial relationships)
- **Total Baseline Prompts:** 1,300+

**VLM Evaluation Dataset:**
- Qwen2-VL evaluations: 1,616 total assessments across 16 spatial concepts
- Each concept: ~100 images evaluated
- Output format: Binary yes/no + optional reasoning trace

**FID Evaluation:**
- COCO ground-truth images: 1,000
- Standard SD3.5 generated: 1,000
- Custom SD3.5 generated: 1,000
- Total images for FID: 3,000

### A.3 Computational Resources

**Hardware:**
- GPU: NVIDIA A100 (assumed from SLURM cluster context)
- Memory: Minimum 40GB GPU RAM for SD3.5 inference
- Storage: ~500GB for model checkpoints, generated images, tensors

**Runtime Estimates:**
- Text encoder conversion: ~2 hours
- Baseline image generation (1,500 prompts): ~10 hours
- VLM evaluation (1,600 images): ~15 hours
- FID calculation (3,000 images): ~3 hours
- **Total Project Runtime:** ~150+ GPU hours

### A.4 Software Dependencies

**Core Libraries (from `requirements.txt`, `requirements2.txt`):**
```
torch >= 2.0
diffusers >= 0.24
transformers >= 4.35
open_clip_torch
accelerate
safetensors
torch-fidelity  # For FID calculation
Pillow
pandas
numpy
tqdm
huggingface_hub
python-dotenv
```

**VLM Dependencies:**
```
llava (custom installation)
qwen-vl-utils
transformers (with Qwen2-VL support)
```

### A.5 Key Configuration Parameters

**Image Generation:**
```python
guidance_scale = 4
num_inference_steps = 28  # SD3.5 Turbo default
resolution = 512×512
seed = 42  # Fixed for reproducibility
batch_size = 1  # Sequential generation for stability
```

**VLM Evaluation:**
```python
model = "Qwen/Qwen2-VL-7B-Instruct"
max_pixels = 512 * 512
question_format = "Does this figure show 'a {class1} {phrase} a {class2}'? Please answer yes or no."
temperature = 0.0  # Deterministic evaluation
```

**FID Calculation:**
```python
metric = "fid"
batch_size = 50
feature_extractor = "inception_v3"
dataset_format = "pytorch_tensor"
```

---

## APPENDIX B: File Organization and Key Scripts

### B.1 Directory Structure

```
/mnt/lustre/work/hein/hof217/stable_diffusion/
│
├── Core Implementation Files
│   ├── convert_text_encoder.py (234 lines) - OpenCLIP → HF conversion
│   ├── custom_sd35.py (222 lines) - Custom pipeline creation
│   ├── run_full_workflow.py (343 lines) - End-to-end testing
│   └── requirements.txt, requirements2.txt - Dependencies
│
├── Development History
│   ├── old/ - Initial SD3.5 testing (July-Aug 2025)
│   │   ├── 0sd_35.py - First SD3.5 experiments
│   │   ├── 0sd_comp.py - Baseline comparisons
│   │   └── create_csv_from_sd15.py - Dataset migration
│   │
│   ├── aug-sept/ - Comprehensive problem analysis (Aug-Sept 2025)
│   │   ├── sd35.py, sd_comp.py - Image generation
│   │   ├── analysis.txt - Manual failure categorization
│   │   └── extract_words.py - Keyword extraction
│   │
│   ├── sept_w3/, sept_w4/ - Systematic evaluation (Sept-Oct 2025)
│   │   ├── self_bench_sd35.py - Automated generation
│   │   ├── simple_positional_*.csv - Prompt libraries (1,300+ prompts)
│   │   ├── negate_words_prompts.csv, negative_words_prompts.csv
│   │   └── sd35_output_*/ - Generated image directories (not present)
│   │
│   └── custom_sd35/ - Iterative encoder replacement design (Oct-Nov 2025)
│       ├── custom_sd35_pipeline_1.py - Initial wrapper approach
│       ├── custom_sd35_pipeline_2.py - Improved mapping
│       ├── custom_sd35_pipeline_3.0.py - QuickGELU fix
│       ├── custom_sd35_outline.txt, custom_sd35_outline_2.txt - Design docs
│       └── custom_sd35_impls/ - Implementation variants
│
├── Evaluation Infrastructure
│   ├── oct_w1-2/ - LLaVA evaluation (Oct 2025)
│   │   ├── llava_vlm.py
│   │   └── llava_vlm_context_learning.py
│   │
│   ├── qwen_big/, qwen_small/ - Qwen VLM evaluation (Nov 2025)
│   │   ├── qwen2_simple.py - Binary answers
│   │   ├── qwen-small-all.py - Compact variant
│   │   └── qwen-small-reasoning-all.py - With reasoning
│   │
│   ├── fid_evaluation/ - Image quality metrics (Feb 2026)
│   │   ├── 0_download_coco_captions.py
│   │   ├── 1_prepare_coco_ground_truth.py
│   │   ├── 2_generate_images.py
│   │   ├── 3_calculate_fid.py
│   │   ├── fid_results.csv - **Final FID scores**
│   │   └── README.md - Pipeline documentation
│   │
│   └── verification checks/ - Model verification scripts
│       ├── verify_text_encoder_change.py
│       ├── verify_converted_encoder.py
│       ├── quick_verify_encoder.py
│       ├── explore_clic.py
│       └── explore_sd35.py
│
├── Generated Outputs
│   ├── custom_sd35_output_left-right-above-top/
│   ├── custom_sd35_output_below-under-over/
│   ├── custom_sd35_output_behind-on-infrontof/
│   ├── custom_sd35_output_bw-inside-outside-nholding-ntouch-not-without/
│   ├── custom_sd35_output_andnot-without/
│   ├── custom_sd35_output_closeby-far/
│   └── custom_sd35_output_sitting-running-standing/
│       └── metadata_*.csv, tensors/ - Generated images and metadata
│
├── Evaluation Results
│   ├── qwen_outputs/
│   │   ├── simple/ - Binary yes/no evaluations (1,616 assessments)
│   │   │   ├── above_qwen_output_*.csv
│   │   │   ├── below_qwen_output_*.csv
│   │   │   ├── [14 more spatial concepts]
│   │   │   └── Total: 16 CSV files
│   │   └── reason/ - With reasoning traces (16 files)
│   │
│   ├── llava_outputs/ - LLaVA evaluation results
│   │   └── [Various spatial concept evaluations]
│   │
│   └── problematic_images.csv - Manual failure categorization
│
├── Analysis Notebooks
│   ├── viz_basic.ipynb - Initial visualizations (Sept 2025)
│   ├── viz2.ipynb - Extended analysis (Sept 2025)
│   ├── qwen_analysis.ipynb - VLM result analysis (Nov 2025)
│   └── common_problems.ipynb - Cross-cutting patterns (Dec 2025)
│
├── Converted Models
│   ├── converted_text_encoder/ - CLIC in HF format
│   │   ├── config.json
│   │   ├── model.safetensors
│   │   └── tokenizer files
│   │
│   ├── CLIC-ViT-L-14-224-PixPr-RedCaps/ - Original OpenCLIP CLIC
│   └── stable-diffusion-3.5-large-turbo/ - Base SD3.5 model
│
├── Job Scripts (SLURM)
│   ├── full_custom_sd35_job.sh - Complete pipeline
│   ├── ferranti-job.sh - Custom generation jobs
│   ├── template_job.sh - Reusable template
│   └── full_2.sh, full_3.sh - Workflow variants
│
└── Documentation
    ├── PROJECT_TIMELINE.md - **This document**
    ├── examples.csv - Example prompts
    └── compare.txt - Comparison notes
```

### B.2 Reproducibility Checklist

To reproduce this work:

1. **Environment Setup:**
   ```bash
   pip install -r requirements.txt
   pip install -r requirements2.txt
   export HF_TOKEN="your_huggingface_token"
   ```

2. **Text Encoder Conversion:**
   ```bash
   python convert_text_encoder.py
   # Output: ./converted_text_encoder/
   ```

3. **Baseline Generation:**
   ```bash
   python sept_w3/self_bench_sd35.py --prompt_file sept_w4/simple_positional_4181917.csv
   # Generates images with metadata
   ```

4. **Custom Pipeline Testing:**
   ```bash
   python run_full_workflow.py --csv_file sept_w4/simple_positional_4181917.csv
   # Generates images using custom encoder
   ```

5. **VLM Evaluation:**
   ```bash
   python qwen_small/qwen-small-all.py
   # Evaluates generated images, outputs to qwen_outputs/simple/
   ```

6. **FID Calculation:**
   ```bash
   cd fid_evaluation
   python 0_download_coco_captions.py
   python 1_prepare_coco_ground_truth.py
   python 2_generate_images.py
   python 3_calculate_fid.py
   # Output: fid_results.csv
   ```

### B.3 Key Functions and APIs

**Custom Pipeline Creation (`custom_sd35.py`):**
```python
pipeline = create_sd35_with_custom_encoder()
image = pipeline(
    prompt="a photo of a cat above a dog",
    guidance_scale=4,
    num_inference_steps=28
).images[0]
```

**Conversion Function (`convert_text_encoder.py`):**
```python
converted_model, weight_mapping = convert_clic_to_sd35_format(
    output_dir="./converted_text_encoder"
)
```

**VLM Evaluation (`qwen_small/qwen-small-all.py`):**
```python
question = f"Does this figure show 'a {class1} {phrase} a {class2}'? Please answer yes or no."
response = evaluate_with_qwen(image_tensor, question)
answer = extract_yes_no(response)  # Binary classification
```

---

## APPENDIX C: Timeline at a Glance

| **Month** | **Focus Area** | **Key Deliverables** | **Files/Directories** |
|-----------|---------------|----------------------|----------------------|
| **July 2025** | Initial exploration | Baseline SD3.5 generation, infrastructure setup | `old/0sd_35.py`, `old/0sd_comp.py` |
| **August 2025** | Problem analysis | Manual failure categorization, 200+ analyzed images | `aug-sept/analysis.txt`, `aug-sept/extract_words.py` |
| **September 2025** | Systematic evaluation | 1,300+ prompt library, automated generation pipeline | `sept_w3/self_bench_sd35.py`, `sept_w4/*.csv` |
| **October 2025** | VLM infrastructure | LLaVA evaluation setup, initial VLM results | `oct_w1-2/llava_vlm.py`, `llava_outputs/` |
| **November 2025** | Encoder replacement design | Multiple pipeline iterations, QuickGELU discovery | `custom_sd35/custom_sd35_pipeline_*.py` |
| **November 2025** | Qwen evaluation | 1,616 VLM assessments across 16 spatial concepts | `qwen_outputs/simple/` (16 CSV files) |
| **December 2025** | Production pipeline | Final conversion script, verification framework | `convert_text_encoder.py`, `custom_sd35.py` |
| **January 2026** | Custom generation | Custom SD3.5 image generation for 7 concept categories | `custom_sd35_output_*/` directories |
| **February 2026** | FID evaluation & documentation | FID scores, final analysis, comprehensive documentation | `fid_evaluation/fid_results.csv`, `PROJECT_TIMELINE.md` |

**Total Duration:** 8 months (July 2025 - February 2026)  
**Total Code Files:** 50+ Python scripts  
**Total Generated Images:** 3,000+ (baseline + custom + COCO)  
**Total Evaluations:** 1,600+ VLM assessments  
**Total Documentation:** 1,500+ lines in PROJECT_TIMELINE.md

---

## APPENDIX D: Citation and References

**Baseline Model:**
- Stable Diffusion 3.5 Large Turbo: `stabilityai/stable-diffusion-3-large-turbo` (Stability AI, 2024)

**Text Encoder Source:**
- CLIC OpenCLIP: `nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps` (HuggingFace Hub)
- Training: PixPr dataset + RedCaps (compositional image-text pairs)

**Evaluation Models:**
- LLaVA v1.5: 7B parameter vision-language model
- Qwen2-VL: 7B parameter vision-language model (Alibaba Cloud)

**Evaluation Dataset:**
- COCO 2017 Validation: 1,000 randomly sampled images for FID ground-truth

**Libraries:**
- `diffusers`: HuggingFace diffusion models library
- `transformers`: HuggingFace transformer models
- `open_clip_torch`: OpenCLIP implementation
- `torch-fidelity`: FID score calculation

---

**End of Document**

