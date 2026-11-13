import os
import torch
import datetime
import io
import argparse
import pandas as pd
from PIL import Image
from pathlib import Path
from tqdm import tqdm
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

# --- CONFIGURATION ---

# Mapping SD model types to their respective base directories
SD_MODEL_PATHS = {
    "standard": "../x5.5-prompts_sd35_output/standard_sd35_output/",
    "redcaps": "../z0-custom_sd35_clic/custom_sd35_output/",
    "cogvlm": "../z4-cogvlm_custom_sd35/cogvlm_sd35_output/"
}

TENSOR_DIRS = [
    'behind_on_front', 'between_inside_outside', 'holding_touching_not',
    'not_without', 'top', 'below_over_under', 'closeby_far',
    'left_right_above', 'sitting_running_standing'
]

PHRASE_INVERSES = {
    "above": "below", "below": "above",
    "to the left of": "to the right of", "to the right of": "to the left of",
    "in front of": "behind", "behind": "in front of",
    "over": "under", "under": "over",
    "inside": "outside", "outside": "inside"
}

STRATEGIES = {
    "single_class": {
        "expert": (
            "You are an Expert Visual Analyst and expert at spatial reasoning.\n"
            "Task: Determine if the {c1} is {phrase}.\n"
            "Step 1: Locate the {c1} in the image and describe its appearance.\n"
            "Step 2: Analyze the specific state or action of the {c1}.\n"
            "Conclusion: [Yes/No/Ambiguous]"
        ),
        "direct": "Does this image show a {c1} {phrase}? Answer Yes, No, or Ambiguous."
    },
    "between": {
        "expert": (
            "You are an expert at spatial reasoning. Task: Is the {c1} between the {c2}?\n"
            "Step 1: Identify and locate the {c1}.\n"
            "Step 2: Identify the group of {c2} or the boundary objects.\n"
            "Step 3: Analyze if the {c1} is positioned in the intermediate space separating the {c2}.\n"
            "Conclusion: [Yes/No/Ambiguous]"
        ),
        "direct": "Is the {c1} located between the {c2}? Answer Yes, No, or Ambiguous."
    },
    "dual_class": {
        "inverse": (
            "You are an Expert Visual Analyst and expert at spatial reasoning.\n"
            "1. Is the {c2} {inverse_phrase} the {c1}?\n"
            "2. If so, then the {c1} is {phrase} the {c2}.\n"
            "Conclusion: [Yes/No/Ambiguous]"
        ),
        "expert": (
            "You are an Expert Visual Analyst and expert at spatial reasoning.\n"
            "1. Identify the {c1} and describe its position.\n"
            "2. Identify the {c2} and describe its position.\n"
            "3. Based on their relative coordinates, is the {c1} {phrase} the {c2}?\n"
            "Conclusion: [Yes/No/Ambiguous]"
        ),
        "direct": "Is the {c1} {phrase} the {c2}? Answer Yes, No, or Ambiguous."
    }
}

# --- HELPERS ---

def get_smart_prompt(row, phrase, strategy_type):
    c1 = row.get('class1')
    c2 = row.get('class2')
    
    if "between" in phrase.lower():
        mode = "between"
        template = STRATEGIES[mode].get(strategy_type, STRATEGIES[mode]["direct"])
        return template.format(c1=c1, c2=c2, phrase=phrase)
    elif pd.isna(c2) or c2 is None or c2 == "":
        mode = "single_class"
        template = STRATEGIES[mode].get(strategy_type, STRATEGIES[mode]["direct"])
        return template.format(c1=c1, phrase=phrase)
    else:
        mode = "dual_class"
        inv = PHRASE_INVERSES.get(phrase, f"not {phrase}")
        # Note: Fixed the key reference to 'expert' as defined in your dictionary
        template = STRATEGIES[mode].get(strategy_type, STRATEGIES[mode]["direct"])
        return template.format(c1=c1, c2=c2, phrase=phrase, inverse_phrase=inv)

def load_tensor_as_image(tensor_path, tensor_index):
    try:
        tensor_data = torch.load(tensor_path, map_location='cpu')
        img_tensor = tensor_data['images'][tensor_index]
        img_tensor = img_tensor.permute(1, 2, 0)
        img = Image.fromarray((img_tensor.cpu().numpy() * 255).astype('uint8'))
        return img
    except Exception as e:
        print(f"Error loading tensor {tensor_path}: {e}")
        return None

def extract_label(text):
    text = text.upper()
    # if after conclusions we have yes or no in any form, we take that as the label. 
    if "CONCLUSION" in text:
        conclusion_part = text.split("CONCLUSION")[-1]
        if "YES" in conclusion_part:
            return "YES"
        elif "NO" in conclusion_part:
            return "NO"
        else:
            return "AMBIGUOUS"


# --- MAIN ---

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("phrase", type=str, help="Phrase to process")
    parser.add_argument("--sd_model", type=str, required=True, choices=["standard", "redcaps", "cogvlm"],
                        help="The type of Stable Diffusion model output to analyze")
    parser.add_argument("--strategies", nargs="+", default=["expert", "direct", "inverse"], 
                        help="Strategies to try")
    args = parser.parse_args()

    # Model Setup
    print(f"Loading Qwen3-VL-30B for analysis of {args.sd_model}...")
    model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
        "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
    )
    processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # Path Resolution
    base_path = SD_MODEL_PATHS[args.sd_model]
    all_metadata = []
    found_dirs = []

    for td in TENSOR_DIRS:
        full_path = os.path.join(base_path, td)
        if os.path.exists(full_path):
            found_dirs.append(full_path)
            csv_files = list(Path(full_path).glob("metadata*.csv"))
            for cf in csv_files:
                df = pd.read_csv(cf)
                matches = df[df['phrase'] == args.phrase]
                if not matches.empty:
                    all_metadata.append(matches)
    
    if not all_metadata:
        print(f"No samples found for phrase '{args.phrase}' in {args.sd_model} data.")
        return
    
    phrase_data = pd.concat(all_metadata, ignore_index=True)

    # Dictionary to hold results per strategy
    strategy_results = {strat: [] for strat in args.strategies}

    for _, row in tqdm(phrase_data.iterrows(), total=len(phrase_data), desc=f"Analyzing {args.phrase}"):
        tensor_path = None
        for d in found_dirs:
            p = os.path.join(d, "tensors", row['tensor_file'])
            if os.path.exists(p):
                tensor_path = p
                break
        
        if not tensor_path: continue
        image = load_tensor_as_image(tensor_path, int(row['tensor_index']))
        if image is None: continue

        for strat in args.strategies:
            prompt_text = get_smart_prompt(row, args.phrase, strat)
            
            messages = [{
                "role": "user",
                "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": image}]
            }]

            inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
            
            with torch.no_grad():
                generated_ids = model.generate(**inputs, max_new_tokens=256)
            
            full_answer = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            
            strategy_results[strat].append({
                'sd_model_type': args.sd_model,
                'tensor_file': row['tensor_file'],
                'tensor_index': row['tensor_index'],
                'phrase': args.phrase,
                'strategy': strat,
                'full_output': full_answer.strip().replace('\n', ' '),
                'predicted_label': extract_label(full_answer),
                'class1': row['class1'],
                'class2': row.get('class2', '')
            })

    # --- NEW SAVING LOGIC ---
    ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    safe_phrase = args.phrase.replace(" ", "_")

    for strat, data in strategy_results.items():
        if not data:
            continue
            
        # Create folder structure: new_qwen_outputs / {sd_model} / {strategy}
        output_dir = os.path.join("new_qwen_outputs", args.sd_model, strat)
        os.makedirs(output_dir, exist_ok=True)
        
        output_filename = f"qwen_{safe_phrase}_{ts}.csv"
        output_path = os.path.join(output_dir, output_filename)
        
        pd.DataFrame(data).to_csv(output_path, index=False)
        print(f"Saved {strat} results to: {output_path}")

if __name__ == "__main__":
    main()

# def main():
#     parser = argparse.ArgumentParser()
#     parser.add_argument("phrase", type=str, help="Phrase to process")
#     parser.add_argument("--sd_model", type=str, required=True, choices=["standard", "redcaps", "cogvlm"],
#                         help="The type of Stable Diffusion model output to analyze")
#     parser.add_argument("--strategies", nargs="+", default=["expert", "direct", "inverse"], 
#                         help="Strategies to try")
#     args = parser.parse_args()

#     # Model Setup
#     print(f"Loading Qwen3-VL-30B for analysis of {args.sd_model}...")
#     model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
#         "Qwen/Qwen3-VL-30B-A3B-Instruct", dtype="auto", device_map="auto"
#     )
#     processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")
#     device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

#     # Path Resolution based on sd_model type
#     base_path = SD_MODEL_PATHS[args.sd_model]
#     all_metadata = []
#     found_dirs = []

#     print(f"Searching in: {base_path}")
#     for td in TENSOR_DIRS:
#         full_path = os.path.join(base_path, td)
#         if os.path.exists(full_path):
#             found_dirs.append(full_path)
#             csv_files = list(Path(full_path).glob("metadata*.csv"))
#             for cf in csv_files:
#                 df = pd.read_csv(cf)
#                 # Filter for the specific phrase
#                 matches = df[df['phrase'] == args.phrase]
#                 if not matches.empty:
#                     all_metadata.append(matches)
    
#     if not all_metadata:
#         print(f"No samples found for phrase '{args.phrase}' in {args.sd_model} data.")
#         return
    
#     phrase_data = pd.concat(all_metadata, ignore_index=True)
#     results = []

#     for _, row in tqdm(phrase_data.iterrows(), total=len(phrase_data), desc=f"Analyzing {args.phrase}"):
#         tensor_path = None
#         for d in found_dirs:
#             p = os.path.join(d, "tensors", row['tensor_file'])
#             if os.path.exists(p):
#                 tensor_path = p
#                 break
        
#         if not tensor_path: continue
#         image = load_tensor_as_image(tensor_path, int(row['tensor_index']))
#         if image is None: continue

#         for strat in args.strategies:
#             prompt_text = get_smart_prompt(row, args.phrase, strat)
            
#             messages = [{
#                 "role": "user",
#                 "content": [{"type": "text", "text": prompt_text}, {"type": "image", "image": image}]
#             }]

#             inputs = processor.apply_chat_template(messages, tokenize=True, add_generation_prompt=True, return_dict=True, return_tensors="pt").to(device)
            
#             with torch.no_grad():
#                 generated_ids = model.generate(**inputs, max_new_tokens=256)
            
#             full_answer = processor.batch_decode(generated_ids[:, inputs["input_ids"].shape[1]:], skip_special_tokens=True)[0]
            
#             results.append({
#                 'sd_model_type': args.sd_model,
#                 'tensor_file': row['tensor_file'],
#                 'tensor_index': row['tensor_index'],
#                 'phrase': args.phrase,
#                 'strategy': strat,
#                 'full_output': full_answer.strip().replace('\n', ' '),
#                 'predicted_label': extract_label(full_answer),
#                 'class1': row['class1'],
#                 'class2': row.get('class2', '')
#             })

#     # Save Output with specific naming convention
#     os.makedirs("new_qwen_outputs", exist_ok=True)
#     ts = datetime.datetime.now().strftime("%Y%m%d_%H%M")
    
#     # Filename includes: phrase, sd_model, strategies, and timestamp
#     strat_str = "_".join(args.strategies)
#     safe_phrase = args.phrase.replace(" ", "_")
#     output_filename = f"qwen_{args.sd_model}_{safe_phrase}_{strat_str}_{ts}.csv"
#     output_path = os.path.join("new_qwen_outputs", output_filename)
    
#     pd.DataFrame(results).to_csv(output_path, index=False)
#     print(f"Analysis complete. Results saved to {output_path}")