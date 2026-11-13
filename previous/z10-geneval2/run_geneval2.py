#!/usr/bin/env python3
"""Evaluate SD3.5 variants on GenEval2 (Soft-TIFA AM + GM).

Variants: standard, clic, cogvlm
Fully resumable: skips existing images and already-scored prompts.
"""

import argparse
import json
from pathlib import Path

import torch
import torch.nn as nn
from diffusers import (
    AutoencoderKL,
    FlowMatchEulerDiscreteScheduler,
    SD3Transformer2DModel,
    StableDiffusion3Pipeline,
)
from scipy.stats import gmean
from tqdm import tqdm
from transformers import (
    AutoProcessor,
    CLIPTextModelWithProjection,
    CLIPTokenizer,
    Qwen3VLForConditionalGeneration,
    T5EncoderModel,
    T5TokenizerFast,
)

WORKSPACE_DIR = Path(__file__).resolve().parent.parent
SD35_DIR = WORKSPACE_DIR / "stable-diffusion-3.5-large-turbo"
CLIC_ENCODER_DIR = WORKSPACE_DIR / "z0-custom_sd35_clic" / "converted_text_encoder"
COGVLM_ENCODER_DIR = WORKSPACE_DIR / "z4-cogvlm_custom_sd35" / "converted_text_encoder_CogVLM"
BENCHMARK_DATA = Path(__file__).resolve().parent / "GenEval2" / "geneval2_data.jsonl"
WORD_TO_NUM = {
    "one": "1", "two": "2", "three": "3", "four": "4", "five": "5",
    "six": "6", "seven": "7", "eight": "8", "nine": "9", "ten": "10",
}


# ── Pipeline building ─────────────────────────────────────────────────────────

class QuickGELU(nn.Module):
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * torch.sigmoid(1.702 * x)


def patch_quickgelu(model):
    for layer in model.text_model.encoder.layers:
        if hasattr(layer.mlp, "activation_fn"):
            layer.mlp.activation_fn = QuickGELU()
    return model


def build_pipeline(variant: str, device: str) -> StableDiffusion3Pipeline:
    if variant == "standard":
        print("Loading standard SD3.5")
        return StableDiffusion3Pipeline.from_pretrained(
            SD35_DIR, torch_dtype=torch.float16
        ).to(device)

    encoder_dir = CLIC_ENCODER_DIR if variant == "clic" else COGVLM_ENCODER_DIR
    print(f"Loading {variant} SD3.5 with encoder from {encoder_dir}")
    pipe = StableDiffusion3Pipeline(
        transformer=SD3Transformer2DModel.from_pretrained(
            SD35_DIR, subfolder="transformer", torch_dtype=torch.float16, device_map=device
        ),
        vae=AutoencoderKL.from_pretrained(
            SD35_DIR, subfolder="vae", torch_dtype=torch.float16, device_map=device
        ),
        scheduler=FlowMatchEulerDiscreteScheduler.from_pretrained(
            SD35_DIR, subfolder="scheduler"
        ),
        text_encoder=patch_quickgelu(CLIPTextModelWithProjection.from_pretrained(
            encoder_dir, torch_dtype=torch.float16, device_map=device
        )),
        text_encoder_2=CLIPTextModelWithProjection.from_pretrained(
            SD35_DIR, subfolder="text_encoder_2", torch_dtype=torch.float16, device_map=device
        ),
        text_encoder_3=T5EncoderModel.from_pretrained(
            SD35_DIR, subfolder="text_encoder_3", torch_dtype=torch.float16, device_map=device
        ),
        tokenizer=CLIPTokenizer.from_pretrained(encoder_dir),
        tokenizer_2=CLIPTokenizer.from_pretrained(SD35_DIR, subfolder="tokenizer_2"),
        tokenizer_3=T5TokenizerFast.from_pretrained(SD35_DIR, subfolder="tokenizer_3"),
    )
    return pipe


# ── Image generation ──────────────────────────────────────────────────────────

def generate_images(pipe, prompts, image_dir, seed, steps, guidance, device):
    image_dir.mkdir(parents=True, exist_ok=True)
    image_map = {}
    for i, entry in enumerate(tqdm(prompts, desc="Generating")):
        prompt = entry["prompt"]
        image_path = image_dir / f"{i:04d}_seed{seed + i}.png"
        if not image_path.exists():
            gen = torch.Generator(device=device).manual_seed(seed + i)
            with torch.inference_mode():
                pipe(
                    prompt=prompt, num_inference_steps=steps,
                    guidance_scale=guidance,
                    height=1024, width=1024, generator=gen,
                ).images[0].save(image_path)
        image_map[prompt] = str(image_path.resolve())
    return image_map


# ── Evaluation (Soft-TIFA) ────────────────────────────────────────────────────

def load_qwen():
    print("Loading Qwen3-VL-8B-Instruct")
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-8B-Instruct")
    processor.tokenizer.padding_side = "left"  # required for correct batched decoding
    model = Qwen3VLForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-8B-Instruct", torch_dtype="auto", device_map="auto"
    )
    # Debug: show where model shards/parameters were placed
    try:
        print("Qwen hf_device_map:", getattr(model, "hf_device_map", None))
    except Exception:
        pass
    try:
        param_devices = sorted({str(p.device) for p in model.parameters()})
        print("Qwen parameter devices:", param_devices)
    except Exception:
        pass
    return processor, model


def score_prompt(processor, model, vqa_list, image_path):
    # Build all questions and answer lists for this image
    questions, answer_lists = [], []
    for question, answer in vqa_list:
        ans_list = (
            [answer, answer.capitalize(), " " + answer, " " + answer.capitalize(),
             WORD_TO_NUM.get(answer, "other"), " " + WORD_TO_NUM.get(answer, "other")]
            if question.startswith("How many")
            else ["Yes", "yes", " yes", " Yes"]
        )
        questions.append(f"{question} Answer in one word.")
        answer_lists.append(ans_list)

    # Batch all questions for this image into one forward pass
    messages_batch = [
        [{"role": "user", "content": [
            {"type": "image", "image": image_path},
            {"type": "text", "text": q},
        ]}]
        for q in questions
    ]
    inputs = processor.apply_chat_template(
        messages_batch, tokenize=True, padding=True,
        add_generation_prompt=True, return_dict=True, return_tensors="pt",
    )

    # Move inputs to a device that matches model parameter placement.
    # Prefer any CUDA device if model has CUDA parameters, otherwise CPU.
    try:
        param_devices = {p.device for p in model.parameters()}
        target_device = next((d for d in param_devices if str(d).startswith("cuda")), None)
        if target_device is None:
            target_device = torch.device("cpu")
    except Exception:
        target_device = torch.device("cpu")

    def _move(batch, device):
        if hasattr(batch, "to"):
            try:
                return batch.to(device)
            except Exception:
                # Fall back to dict-style move
                pass
        return {k: v.to(device) for k, v in batch.items()}

    inputs = _move(inputs, target_device)

    with torch.inference_mode():
        out = model.generate(
            **inputs, max_new_tokens=1, do_sample=False,
            output_scores=True, return_dict_in_generate=True,
        )

    # out.scores[0] shape: (batch_size, vocab_size)
    probs = torch.nn.functional.softmax(out.scores[0], dim=-1)
    return [
        sum(probs[i, processor.tokenizer.encode(a)[0]].item() for a in ans_list)
        for i, ans_list in enumerate(answer_lists)
    ]


def evaluate(prompts, image_map, scores_file, processor, model):
    scores = json.loads(scores_file.read_text()) if scores_file.exists() else {}

    for entry in tqdm(prompts, desc="Evaluating"):
        prompt = entry["prompt"]
        if prompt in scores:
            continue
        scores[prompt] = score_prompt(processor, model, entry["vqa_list"], image_map[prompt])
        scores_file.write_text(json.dumps(scores))  # save after every prompt

    all_lists = [scores[e["prompt"]] for e in prompts]
    am = 100 * sum(sum(s) / len(s) for s in all_lists) / len(all_lists)
    gm = 100 * sum(gmean(s) for s in all_lists) / len(all_lists)
    return am, gm


# ── Main ──────────────────────────────────────────────────────────────────────

def main():
    p = argparse.ArgumentParser()
    p.add_argument("--model_variant", choices=["standard", "clic", "cogvlm"], required=True)
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--max_prompts", type=int, default=None,
                   help="Use only the first N prompts (for quick tests)")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--steps", type=int, default=28)
    p.add_argument("--guidance", type=float, default=7.0)
    args = p.parse_args()

    output_dir = args.output_dir or Path(__file__).resolve().parent / "outputs" / args.model_variant
    output_dir.mkdir(parents=True, exist_ok=True)
    image_map_path = output_dir / "image_filepath_data.json"
    scores_file = output_dir / "scores.json"

    prompts = [json.loads(l) for l in BENCHMARK_DATA.open() if l.strip()]
    if args.max_prompts:
        prompts = prompts[: args.max_prompts]
    print(f"Variant={args.model_variant}  prompts={len(prompts)}  output={output_dir}")

    # Generation (skips existing images; saves map when all done)
    if image_map_path.exists():
        image_map = json.loads(image_map_path.read_text())
        print(f"Found existing image map — skipping generation")
    else:
        device = "cuda" if torch.cuda.is_available() else "cpu"
        pipe = build_pipeline(args.model_variant, device)
        image_map = generate_images(pipe, prompts, output_dir / "images", args.seed, args.steps, args.guidance, device)
        del pipe
        torch.cuda.empty_cache()
        image_map_path.write_text(json.dumps(image_map, indent=2))
        print(f"Saved image map → {image_map_path}")

    # Evaluation (resumes from scores.json if interrupted)
    processor, model = load_qwen()
    am, gm = evaluate(prompts, image_map, scores_file, processor, model)

    results = {
        "variant": args.model_variant,
        "num_prompts": len(prompts),
        "soft_tifa_am": round(am, 4),
        "soft_tifa_gm": round(gm, 4),
    }
    (output_dir / "results.json").write_text(json.dumps(results, indent=2))

    print(f"\n{'─'*40}")
    print(f"Soft-TIFA AM : {am:.2f}")
    print(f"Soft-TIFA GM : {gm:.2f}")
    print(f"{'─'*40}")
    print(f"Saved → {output_dir / 'results.json'}")


if __name__ == "__main__":
    main()
