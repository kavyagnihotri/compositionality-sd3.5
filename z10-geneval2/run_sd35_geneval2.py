#!/usr/bin/env python3
"""Generate GenEval2 images with the local SD3.5 checkpoint and evaluate them.

This script follows the GenEval2 README flow:
1. Generate one image per prompt from geneval2_data.jsonl.
2. Save a prompt -> image filepath mapping JSON for evaluation.py.
3. Run Soft-TIFA AM, Soft-TIFA GM, and the per-skill analysis script.

The default model path points at the local stable-diffusion-3.5-large-turbo
checkpoint in this workspace.
"""

from __future__ import annotations

import argparse
import json
import subprocess
import sys
from pathlib import Path

import torch
from diffusers import StableDiffusion3Pipeline
from PIL import Image
from tqdm import tqdm


SCRIPT_DIR = Path(__file__).resolve().parent
WORKSPACE_DIR = SCRIPT_DIR.parent
DEFAULT_MODEL_DIR = WORKSPACE_DIR / "stable-diffusion-3.5-large-turbo"
DEFAULT_BENCHMARK_DATA = SCRIPT_DIR / "GenEval2" / "geneval2_data.jsonl"
DEFAULT_EVALUATION_SCRIPT = SCRIPT_DIR / "GenEval2" / "evaluation.py"
DEFAULT_ANALYSIS_SCRIPT = SCRIPT_DIR / "GenEval2" / "soft_tifa_analysis.py"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Generate and evaluate GenEval2 images with SD3.5")
    parser.add_argument("--model_dir", type=Path, default=DEFAULT_MODEL_DIR, help="Local SD3.5 model directory")
    parser.add_argument("--benchmark_data", type=Path, default=DEFAULT_BENCHMARK_DATA, help="GenEval2 JSONL file")
    parser.add_argument("--evaluation_script", type=Path, default=DEFAULT_EVALUATION_SCRIPT, help="Path to GenEval2 evaluation.py")
    parser.add_argument("--analysis_script", type=Path, default=DEFAULT_ANALYSIS_SCRIPT, help="Path to GenEval2 soft_tifa_analysis.py")
    parser.add_argument("--output_dir", type=Path, default=SCRIPT_DIR / "outputs" / "sd35_geneval2", help="Directory for generated images and scores")
    parser.add_argument("--seed", type=int, default=42, help="Base seed used for generation")
    parser.add_argument("--num_inference_steps", type=int, default=4, help="Inference steps for SD3.5 Turbo")
    parser.add_argument("--guidance_scale", type=float, default=0.0, help="Guidance scale for SD3.5 Turbo")
    parser.add_argument("--height", type=int, default=1024, help="Image height")
    parser.add_argument("--width", type=int, default=1024, help="Image width")
    parser.add_argument("--skip_evaluation", action="store_true", help="Only generate images and mapping JSON")
    parser.add_argument("--skip_analysis", action="store_true", help="Skip the per-skill analysis step")
    return parser.parse_args()


def load_prompts(benchmark_data: Path) -> list[dict]:
    with benchmark_data.open("r", encoding="utf-8") as handle:
        return [json.loads(line) for line in handle if line.strip()]


def build_pipeline(model_dir: Path, device: str) -> StableDiffusion3Pipeline:
    pipe = StableDiffusion3Pipeline.from_pretrained(model_dir, torch_dtype=torch.float16)
    pipe = pipe.to(device)
    return pipe


def generate_images(
    pipe: StableDiffusion3Pipeline,
    prompts: list[dict],
    output_image_dir: Path,
    seed: int,
    guidance_scale: float,
    num_inference_steps: int,
    height: int,
    width: int,
    device: str,
) -> dict[str, str]:
    output_image_dir.mkdir(parents=True, exist_ok=True)
    prompt_to_path: dict[str, str] = {}

    for index, entry in enumerate(tqdm(prompts, desc="Generating GenEval2 images")):
        prompt = entry["prompt"]
        prompt_seed = seed + index
        generator = torch.Generator(device=device).manual_seed(prompt_seed)

        with torch.inference_mode():
            result = pipe(
                prompt=prompt,
                num_inference_steps=num_inference_steps,
                guidance_scale=guidance_scale,
                height=height,
                width=width,
                generator=generator,
            )

        image: Image.Image = result.images[0]
        file_name = f"{index:04d}_seed{prompt_seed}.png"
        image_path = output_image_dir / file_name
        image.save(image_path)
        prompt_to_path[prompt] = str(image_path.resolve())

    return prompt_to_path


def run_subprocess(command: list[str]) -> None:
    print("\nRunning:")
    print(" ".join(command))
    subprocess.run(command, check=True)


def main() -> None:
    args = parse_args()

    if not args.model_dir.exists():
        raise FileNotFoundError(f"Model directory not found: {args.model_dir}")
    if not args.benchmark_data.exists():
        raise FileNotFoundError(f"Benchmark data not found: {args.benchmark_data}")
    if not args.evaluation_script.exists():
        raise FileNotFoundError(f"Evaluation script not found: {args.evaluation_script}")
    if not args.analysis_script.exists():
        raise FileNotFoundError(f"Analysis script not found: {args.analysis_script}")

    device = "cuda" if torch.cuda.is_available() else "cpu"
    output_dir = args.output_dir
    image_dir = output_dir / "images"
    output_dir.mkdir(parents=True, exist_ok=True)

    prompts = load_prompts(args.benchmark_data)
    print(f"Loaded {len(prompts)} GenEval2 prompts from {args.benchmark_data}")
    print(f"Loading SD3.5 from {args.model_dir} on {device}")
    pipe = build_pipeline(args.model_dir, device)

    prompt_to_path = generate_images(
        pipe=pipe,
        prompts=prompts,
        output_image_dir=image_dir,
        seed=args.seed,
        guidance_scale=args.guidance_scale,
        num_inference_steps=args.num_inference_steps,
        height=args.height,
        width=args.width,
        device=device,
    )

    del pipe
    if torch.cuda.is_available():
        torch.cuda.empty_cache()

    image_map_path = output_dir / "image_filepath_data.json"
    with image_map_path.open("w", encoding="utf-8") as handle:
        json.dump(prompt_to_path, handle, indent=2, ensure_ascii=False)
    print(f"Saved image mapping to {image_map_path}")

    if args.skip_evaluation:
        return

    am_scores = output_dir / "soft_tifa_am_score_lists.json"
    gm_scores = output_dir / "soft_tifa_gm_score_lists.json"

    run_subprocess([
        sys.executable,
        str(args.evaluation_script),
        "--benchmark_data",
        str(args.benchmark_data),
        "--image_filepath_data",
        str(image_map_path),
        "--method",
        "soft_tifa_am",
        "--output_file",
        str(am_scores),
    ])

    run_subprocess([
        sys.executable,
        str(args.evaluation_script),
        "--benchmark_data",
        str(args.benchmark_data),
        "--image_filepath_data",
        str(image_map_path),
        "--method",
        "soft_tifa_gm",
        "--output_file",
        str(gm_scores),
    ])

    if args.skip_analysis:
        return

    run_subprocess([
        sys.executable,
        str(args.analysis_script),
        "--benchmark_data",
        str(args.benchmark_data),
        "--score_data",
        str(am_scores),
    ])


if __name__ == "__main__":
    main()