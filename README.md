# Understanding and Improving Compositionality of Stable Diffusion 3.5

Text-to-image models still get simple spatial relations wrong ("a cup **above** a laptop", "a dog **to the left of** a car").
This project measures how often **Stable Diffusion 3.5 Large Turbo** fails at relational / spatial prompts and tests whether
swapping its CLIP-L text encoder for a **compositionally fine-tuned CLIC encoder** helps — without retraining the diffusion model.

Three variants are compared throughout:

| Variant | CLIP-L text encoder (`text_encoder`) | Source |
|---|---|---|
| **Standard** | original OpenAI CLIP-L | `stabilityai/stable-diffusion-3.5-large-turbo` |
| **CLIC RedCaps** | CLIC ViT-L/14, fine-tuned on RedCaps | `nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps` |
| **CLIC CogVLM** | CLIC ViT-L/14, fine-tuned on CogVLM captions | `nmndeep/CLIC-ViT-L-14-224-CogVLM` |

The OpenCLIP-G (`text_encoder_2`), T5-XXL (`text_encoder_3`), MMDiT transformer, VAE and scheduler are left unchanged.

---

## Method in one paragraph

1. **Find failures** – generate images for 16+ relational phrases (above, below, left/right of, between, behind, in front of, on top of, over, under, holding, touching, without, …) from COCO class pairs.
2. **Score them** – a VLM judge (LLaVA first, then Qwen3-VL-30B-A3B) answers Yes / No / Ambiguous per image, using several prompt strategies (*direct*, *expert*, *inverse*) combined by majority vote; 10–20 images per phrase are labelled by hand to check the judge.
3. **Swap the encoder** – convert the OpenCLIP CLIC text tower to a HuggingFace `CLIPTextModelWithProjection` (split fused QKV, rename MLP layers, transpose the text projection, QuickGELU activation) and plug it into the SD3.5 pipeline.
4. **Check quality & compositionality** – FID on COCO val2017 (5k captions) and Soft-TIFA on the GenEval2 benchmark (800 prompts, Qwen3-VL-8B oracle).

## Headline results

**Spatial accuracy (%)** — majority-voted Qwen3-VL scoring, 4-step Turbo:

| Phrase | Standard | RedCaps | CogVLM |
|---|---:|---:|---:|
| above | 72.7 | 85.0 | **90.0** |
| below | 70.0 | 75.0 | **90.0** |
| to the left of | 78.8 | **80.0** | 75.0 |
| to the right of | **84.9** | 65.0 | 80.0 |
| between | **68.0** | – | 55.0 |
| behind | **63.2** | 60.0 | 60.0 |
| in front of | 68.4 | 50.0 | **70.0** |
| on top of | **33.3** | 30.0 | 20.0 |
| over | **70.0** | 40.0 | 40.0 |
| under | **40.0** | **40.0** | 30.0 |
| without | **70.0** | 30.0 | 30.0 |

**GenEval2 Soft-TIFA** (`z10-geneval2/`):

| Model | AM (%) | GM (%) |
|---|---:|---:|
| SD3.5-Large (paper, non-Turbo) | 69.70 | 22.80 |
| Standard SD3.5 Turbo | 55.48 | 8.91 |
| CLIC RedCaps Turbo | 56.34 | **10.68** |
| CLIC CogVLM Turbo | **56.52** | 10.51 |

**FID vs. COCO val2017** (5k images, 512×512, lower is better; `z9-fid_new/fid_results.csv`):

| Model | pytorch-fid | clean-fid |
|---|---:|---:|
| Standard SD3.5 Turbo | 33.84 | 34.70 |
| CLIC RedCaps Turbo | 34.07 | 34.90 |
| CLIC CogVLM Turbo | 34.24 | 35.12 |

All three are within ~0.4 FID of each other, so swapping the encoder does not hurt image quality.

Takeaway: the encoder swap clearly helps vertical relations (above / below) and slightly improves GenEval2, but hurts some others (over, without), so the compositional signal from CLIC is real but uneven.

---

## Repository layout

The numbered folders roughly follow the project timeline. `x*` = early exploration, `z*` = main pipeline.

### Generation & prompts

| Folder | What's in it |
|---|---|
| `x1-different_sd/` | First pilot: SD 1.5 vs SD 3.5 on SugarCrepe++ prompts. |
| `x2-different_sd35/` | SD3.5 from a plain prompt file (`sd35.py --prompt_file ...`). |
| `x4-prompts/` | Hand-written prompt lists used in the pilot. |
| `x4.5-outputs/` | Metadata / analysis from pilot runs (images not tracked). |
| `x5-selfbench/` | Self-Bench-style prompts and generation. |
| `x5.5-prompts_sd35_output/` | **Main baseline generation.** Prompt CSVs built from COCO class pairs (`create_prompts.ipynb`), `sd35_prompt.py` generates one tensor file per phrase into `standard_sd35_output/<group>/`. Also holds `problematic_images.csv` (manual labels). |
| `z1-new-prompts/` | Additional prompt sets (single-class actions: sitting / standing / running; misc: closeby / far from). |

### Text-encoder replacement

| Folder | What's in it |
|---|---|
| `z0-custom_sd35_clic/` | `convert_text_encoder.py` (OpenCLIP → HF conversion for **CLIC RedCaps**), `custom_sd35.py` (`create_sd35_with_custom_encoder()` + QuickGELU patch), `run_full_workflow.py` (generate from prompt CSVs). Converted tokenizer/config in `converted_text_encoder/`. |
| `z4-cogvlm_custom_sd35/` | Same pipeline for **CLIC CogVLM**, plus `compare_encoders.py` and verification scripts. |
| `z0.5-verification_checks/` | Sanity checks that the converted encoder matches the OpenCLIP original (`verify_clic_conversion.py`, `verify_converted_encoder.py`). |

### Evaluation

| Folder | What's in it |
|---|---|
| `x6-llava_vlm/` | First VLM judge (LLaVA), incl. an in-context-learning variant. |
| `x7-z1-qwen_small/` | **Main VLM judge** — Qwen3-VL-30B-A3B. `3_qwen.py` runs the *direct / expert / inverse* prompt strategies; outputs in `new_qwen_outputs/{standard,redcaps,cogvlm}/{direct,expert,inverse}/`. Majority-vote analysis in `qwen_majority_vote_analysis.ipynb`. |
| `z5-manual_annotations/`, `z6/` | Manual labels per variant and Qwen-vs-human confusion matrices (`compare_qwen_manual.ipynb`). |
| `z2-analysis/` | Numbered analysis scripts (evaluator comparison, baseline vs custom, encoder verification) → `analysis_results/`. Has its own README. |
| `z3-fid_evaluation/` | First FID run (1k COCO captions, tensors). Has its own README. |
| `z8-coco/` | Second FID iteration (5k captions). |
| `z9-fid_new/` | **Final FID pipeline** — raw COCO JPGs, turbo-safe `guidance_scale=0`, `pytorch-fid` or `clean-fid`. Has its own README. |
| `z10-geneval2/` | GenEval2 Soft-TIFA evaluation of all three variants (`run_geneval2.py`); scores in `outputs/*/`. Needs the `GenEval2/` repo cloned inside. |
| `z11-ambi-imaes/` | Curated examples of genuinely ambiguous images (`info.md`). |

### Top level

| File | Purpose |
|---|---|
| `viz_basic.ipynb`, `viz2.ipynb` | Browse generated images with prompts and VLM verdicts. |
| `common_problems.ipynb` | Gallery of recurring failure modes. |
| `template_job.sh`, `ferranti-job.sh` | SLURM job templates for the cluster. |
| `requirements.txt`, `requirements2.txt` | Python environments (generation / evaluation). |
| `notes.txt`, `todo.txt` | Working notes, incl. early manual checks per phrase. |
| `previous/` | Earlier version of this repo: LaTeX report (`report/`), result tables & plots (`results/`), helper scripts (`misc/`). |

---

## Setup

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt          # add requirements2.txt for the Qwen3-VL / evaluation stack
echo "HF_TOKEN=hf_..." > .env            # SD3.5 is a gated model
```

Download the models next to the code (they are git-ignored):

```bash
git lfs install
git clone https://huggingface.co/stabilityai/stable-diffusion-3.5-large-turbo
git clone https://huggingface.co/nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps
git clone https://huggingface.co/nmndeep/CLIC-ViT-L-14-224-CogVLM
git clone https://github.com/facebookresearch/GenEval2 z10-geneval2/GenEval2
```

> **Note:** several scripts contain absolute cluster paths (e.g. `LOCAL_SD_DIR` in `z0-custom_sd35_clic/custom_sd35.py`,
> `#SBATCH --output` in the job scripts). Adjust them to your machine before running.

## Typical workflow

```bash
# 1. Convert a CLIC encoder to HF format (RedCaps; use z4-cogvlm_custom_sd35/ for CogVLM)
cd z0-custom_sd35_clic && python convert_text_encoder.py
python ../z0.5-verification_checks/verify_converted_encoder.py

# 2. Generate images
python ../x5.5-prompts_sd35_output/sd35_prompt.py --csv_file <prompts.csv>      # standard SD3.5
python run_full_workflow.py --csv_file <prompts.csv> --phrase above              # custom encoder

# 3. Judge with Qwen3-VL (one phrase per call; --sd_model: standard | redcaps | cogvlm)
cd ../x7-z1-qwen_small
python 3_qwen.py "above" --sd_model cogvlm --strategies expert direct inverse

# 4. Image quality (FID)
cd ../z9-fid_new
python 1_download_captions.py && python 2_coco_ground_truth.py
python 3_generate_images.py --model all
python 4_fid.py                    # pytorch-fid
python 4_fid.py --use-clean-fid    # clean-fid

# 5. GenEval2
cd ../z10-geneval2 && python run_geneval2.py --help
```

On the cluster, wrap each step in a SLURM script based on `template_job.sh`.

## Author

Kavyanjali Agnihotri. Full write-up: `previous/report/report_template.pdf`.
