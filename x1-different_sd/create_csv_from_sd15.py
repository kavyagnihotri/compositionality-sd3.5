import os
import argparse
import pandas as pd
from tqdm import tqdm
from datasets import load_dataset

def get_category_from_path(path):
    parts = path.split(os.sep)
    return parts[-3]  # category is 3 levels up from file

def get_prompt_type_from_path(path):
    parts = path.split(os.sep)
    return parts[-2]  # prompt_type is directory name

def find_all_tensor_files(root_dir):
    tensor_files = []
    for root, _, files in os.walk(os.path.join(root_dir, "tensors", "sd15")):
        for file in files:
            if file.endswith('.pt'):
                tensor_files.append(os.path.join(root, file))
    return tensor_files

def create_prompt_mapping():
    categories = [
        "replace_attribute",
        "replace_object",
        "replace_relation",
        "swap_atribute",
        "swap_object"
    ]
    
    prompt_map = {}
    for category in categories:
        ds = load_dataset("Aman-J/SugarCrepe_pp", category)
        for example_idx, example in enumerate(ds['train']):
            for prompt_type in ['caption', 'caption2', 'negative_caption']:
                key = (category, prompt_type, example_idx)
                prompt_map[key] = example[prompt_type]
    return prompt_map

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tensor_dir", required=True, help="Directory containing SD1.5 tensor outputs")
    parser.add_argument("--output_csv", required=True, help="Path to output metadata CSV file")
    args = parser.parse_args()
    
    # Create prompt mapping from original dataset
    print("Loading SugarCrepe dataset to reconstruct prompts...")
    prompt_map = create_prompt_mapping()
    
    # Find all tensor files
    print("Scanning for tensor files...")
    tensor_files = find_all_tensor_files(args.tensor_dir)
    
    # Generate metadata
    metadata = []
    for tensor_file in tqdm(tensor_files, desc="Processing tensors"):
        try:
            category = get_category_from_path(tensor_file)
            prompt_type = get_prompt_type_from_path(tensor_file)
            index = int(os.path.basename(tensor_file).split('.')[0])
            
            # Get original prompt
            prompt = prompt_map[(category, prompt_type, index)]
            
            # Get relative path to tensor file
            rel_tensor_path = os.path.relpath(tensor_file, args.tensor_dir)
            
            metadata.append({
                'category': category,
                'prompt_type': prompt_type,
                'prompt_text': prompt,
                'tensor_path': rel_tensor_path,
                'image_index': index,
                'model': 'sd15'  # Indicates these came from SD1.5
            })
            
        except Exception as e:
            print(f"Error processing {tensor_file}: {e}")
            continue
    
    # Save metadata
    df = pd.DataFrame(metadata)
    df.to_csv(args.output_csv, index=False)
    print(f"\nSuccessfully created metadata for {len(df)} images")
    print(f"Metadata saved to: {args.output_csv}")

if __name__ == "__main__":
    main()
