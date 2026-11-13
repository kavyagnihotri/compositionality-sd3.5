import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse
from transformers import AutoProcessor, AutoModelForImageTextToText
from tqdm import tqdm
import datetime

# Directories containing tensor data
tensor_dirs = ["sd35_output_behind_on_front", "sd35_output_below_over_under",
               "sd35_output_between_inside_outside", "sd35_output_holding_touching_not", 
               "sd35_output_left_right_above", "sd35_output_top"]
base_path = "./sept_w4" 
tensor_dirs = [os.path.join(base_path, d) for d in tensor_dirs]
timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")

def load_metadata_from_dir(tensor_dir):
    """Load all metadata CSV files from a directory"""
    csv_files = list(Path(tensor_dir).glob("metadata*.csv"))
    all_metadata = []
    
    for csv_file in csv_files:
        try:
            df = pd.read_csv(csv_file)
            all_metadata.append(df)
        except Exception as e:
            print(f"Error loading {csv_file}: {e}")
    
    if all_metadata:
        return pd.concat(all_metadata, ignore_index=True)
    return pd.DataFrame()

def load_tensor_as_image(tensor_path, tensor_index):
    """Load a tensor file as an image"""
    try:
        # Load the tensor file which contains multiple images
        tensor_data = torch.load(tensor_path)
        
        # The tensor is stored in a dictionary with 'images' key
        if isinstance(tensor_data, dict) and 'images' in tensor_data:
            img_tensor = tensor_data['images'][tensor_index]
        else:
            print(f"Unexpected tensor structure in {tensor_path}")
            return None
            
        # Convert tensor to PIL Image
        img_tensor = img_tensor.permute(1, 2, 0)  # CHW -> HWC
        img_tensor = img_tensor * 255  # Scale to 0-255 range for PIL
        img = Image.fromarray(img_tensor.cpu().numpy().astype('uint8'))
        return img
    except Exception as e:
        print(f"Error loading tensor {tensor_path} at index {tensor_index}: {str(e)}")
        return None
    
def create_llava_message(image, class1, phrase, class2):
    question = f"Is a {class1} {phrase} a {class2}? Answer in Yes or No"
    messages = [
        {
            "role": "user",
            "content": [
                {"type": "image", "url": image},
                {"type": "text", "text": question}
            ]
        },
    ]
    return messages, question

def load_examples_csv(examples_path):
    """
    Load the examples CSV file containing tensor_index, phrase, example_type
    Returns: pd.DataFrame: DataFrame containing the examples data
    """
    try:
        examples_df = pd.read_csv(examples_path)
        required_columns = ['tensor_index', 'phrase', 'example_type']
        
        if not all(col in examples_df.columns for col in required_columns):
            missing = [col for col in required_columns if col not in examples_df.columns]
            print(f"Error: Examples CSV missing required columns: {missing}")
            return None
            
        return examples_df
    except Exception as e:
        print(f"Error loading examples CSV: {e}")
        return None

def find_example_in_metadata(tensor_index, phrase, all_metadata, tensor_dirs):
    """
    Find tensor file and metadata for a specific example by matching phrase and tensor_index
    Returns: tuple: (metadata_row, tensor_dir, tensor_file) 
        or (None, None, None) if not found
    """
    for dir_idx, metadata_df in enumerate(all_metadata):
        # Match by phrase and tensor_index
        matches = metadata_df[(metadata_df['phrase'] == phrase) & 
                             (metadata_df['tensor_index'] == tensor_index)]
        
        if not matches.empty:
            metadata_row = matches.iloc[0]
            tensor_dir = tensor_dirs[dir_idx]
            tensor_file = metadata_row['tensor_file']
            return metadata_row, tensor_dir, tensor_file
    
    return None, None, None

def prepare_examples_for_phrase(examples_df, phrase, all_metadata, tensor_dirs):
    """
    Prepare positive and negative examples for a specific phrase
    Returns: list: List of dictionaries with example images and their descriptions
    """
    phrase_examples = examples_df[examples_df['phrase'] == phrase]
    if phrase_examples.empty:
        print(f"No examples found for phrase: '{phrase}'")
        return []
        
    examples = []
    pos_examples = phrase_examples[phrase_examples['example_type'] == 'positive']
    neg_examples = phrase_examples[phrase_examples['example_type'] == 'negative']
    
    # Process both positive and negative examples
    for example_type, type_examples in [('positive', pos_examples), ('negative', neg_examples)]:
        for _, row in type_examples.iterrows():
            tensor_index = int(row['tensor_index'])
            
            # Find metadata and tensor file for this example
            metadata_row, tensor_dir, tensor_file = find_example_in_metadata(
                tensor_index, phrase, all_metadata, tensor_dirs
            )
            
            if metadata_row is None:
                print(f"Warning: No metadata found for example with tensor_index={tensor_index}, phrase='{phrase}'")
                continue
            
            # Construct tensor path
            tensor_path = os.path.join(tensor_dir, "tensors", tensor_file)
            
            if not os.path.exists(tensor_path):
                print(f"Warning: Tensor file not found: {tensor_path}")
                continue
                
            # Load the image from tensor
            image = load_tensor_as_image(tensor_path, tensor_index)
            if image is None:
                continue
                
            # Add to examples
            examples.append({
                'image': image,
                'class1': metadata_row['class1'],
                'class2': metadata_row['class2'],
                'phrase': phrase,
                'example_type': example_type,
                'tensor_index': tensor_index,
                'tensor_file': tensor_file
            })
    
    print(f"Prepared {len(examples)} examples for phrase '{phrase}' "
          f"({len([e for e in examples if e['example_type'] == 'positive'])} positive, "
          f"{len([e for e in examples if e['example_type'] == 'negative'])} negative)")
    
    return examples

def create_incontext_prompt(image, class1, phrase, class2, examples):
    """
    Create an in-context learning prompt for LLaVa using examples
    Args:
        image (PIL.Image): The target image to evaluate
        class1 (str): First class in the relation
        class2 (str): Second class in the relation
        phrase (str): Relation phrase
        examples (list): List of example dictionaries with images and metadata
        
    Returns:
        list: LLaVa messages with in-context examples
        str: The question being asked
    """
    messages = []
    
    # Add examples
    for i, example in enumerate(examples):
        answer = "Yes" if example['example_type'] == 'positive' else "No"
        example_prompt = f"Is a {example['class1']} {example['phrase']} a {example['class2']}?"
        
        messages.append({
            "role": "user",
            "content": [
                {"type": "image"},
                {"type": "text", "text": example_prompt}
            ]
        })
        
        messages.append({
            "role": "assistant", 
            "content": [
                {"type": "text", "text": f"{answer}. The image {'shows' if example['example_type'] == 'positive' else 'does not show'} a {example['class1']} {example['phrase']} a {example['class2']}."}
            ]
        })
    
    # Add the actual query
    question = f"Is a {class1} {phrase} a {class2}? Answer in Yes or No."
    messages.append({
        "role": "user",
        "content": [
            {"type": "image"},
            {"type": "text", "text": question}
        ]
    })
    
    return messages, question

def process_phrase_data_with_incontext(phrase_data, found_dirs, phrase, examples_df, all_metadata):
    """Process data for a specific phrase with in-context learning examples"""
    results = []
    
    print("Loading LLaVa model...")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    examples = prepare_examples_for_phrase(examples_df, phrase, all_metadata, tensor_dirs)
    if not examples:
        print(f"Warning: No examples prepared for phrase '{phrase}'. Using standard prompting.")
    
    for _, row in tqdm(phrase_data.iterrows(), total=len(phrase_data), desc=f"Processing '{phrase}'"):
        tensor_file = row['tensor_file']
        tensor_index = int(row['tensor_index'])
        class1 = row['class1']
        class2 = row['class2']
        prompt = row['prompt']
        
        tensor_path = None
        for tensor_dir in found_dirs:
            potential_path = os.path.join(tensor_dir, "tensors", tensor_file)
            if os.path.exists(potential_path):
                tensor_path = potential_path
                break
        
        if not tensor_path:
            print(f"Tensor file not found: {tensor_file}")
            continue
            
        image = load_tensor_as_image(tensor_path, tensor_index)
        if image is None:
            continue
            
        if examples:
            messages, question = create_incontext_prompt(image, class1, phrase, class2, examples)
        else:
            # Fall back to standard prompting if no examples
            messages, question = create_llava_message(image, class1, phrase, class2)
        
        # Process with LLaVa
        print(messages)
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(**inputs, max_new_tokens=40)
        full_answer = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        # Extract just "Yes" or "No" from the answer
        if "yes" in full_answer.lower():
            clean_answer = "YES"
        elif "no" in full_answer.lower():
            clean_answer = "NO"
        else:
            clean_answer = "Unclear"
        
        # Store result
        results.append({
            'tensor_file': tensor_file,
            'tensor_index': tensor_index,
            'prompt': prompt,
            'class1': class1,
            'class2': class2,
            'phrase': phrase,
            'question': question,
            'full_answer': full_answer,
            'answer': clean_answer,
            'used_incontext': len(examples) > 0,
        })
    
    return results

def find_phrase_in_metadata(phrase, all_metadata):
    """Find all entries for a specific phrase in all metadata"""
    phrase_data = []
    found_dirs = set()
    
    for dir_idx, df in enumerate(all_metadata):
        matches = df[df['phrase'] == phrase]
        if not matches.empty:
            phrase_data.append(matches)
            found_dirs.add(dir_idx)
    
    if phrase_data:
        combined_data = pd.concat(phrase_data, ignore_index=True)
        return combined_data, [tensor_dirs[idx] for idx in found_dirs]
    return None, []

def process_phrase_data(phrase_data, found_dirs, phrase):
    """Process data for a specific phrase"""
    results = []
    
    # Initialize model
    print("Loading LLaVa model...")
    processor = AutoProcessor.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    model = AutoModelForImageTextToText.from_pretrained("llava-hf/llava-v1.6-mistral-7b-hf")
    device = "cuda" if torch.cuda.is_available() else "cpu"
    model = model.to(device)
    print(f"Model loaded on {device}")
    
    # Process each entry for the phrase
    for _, row in tqdm(phrase_data.iterrows(), total=len(phrase_data), desc=f"Processing '{phrase}'"):
        tensor_file = row['tensor_file']
        tensor_index = int(row['tensor_index'])
        class1 = row['class1']
        class2 = row['class2']
        prompt = row['prompt']
        
        # Find the tensor file in one of the found directories
        tensor_path = None
        for tensor_dir in found_dirs:
            potential_path = os.path.join(tensor_dir, "tensors", tensor_file)
            if os.path.exists(potential_path):
                tensor_path = potential_path
                break
        
        if not tensor_path:
            print(f"Tensor file not found: {tensor_file}")
            continue
            
        image = load_tensor_as_image(tensor_path, tensor_index)
        if image is None:
            continue
            
        # Create message and get response
        messages, question = create_llava_message(image, class1, phrase, class2)
        
        # Process with LLaVa
        inputs = processor.apply_chat_template(
            messages,
            add_generation_prompt=True,
            tokenize=True,
            return_dict=True,
            return_tensors="pt",
        ).to(device)

        outputs = model.generate(**inputs, max_new_tokens=40)
        full_answer = processor.decode(outputs[0][inputs["input_ids"].shape[-1]:], skip_special_tokens=True).strip()
        
        # Extract just "Yes" or "No" from the answer
        if "yes" in full_answer.lower():
            clean_answer = "YES"
        elif "no" in full_answer.lower():
            clean_answer = "NO"
        else:
            clean_answer = "Unclear"
        
        # Store result
        results.append({
            'tensor_file': tensor_file,
            'tensor_index': tensor_index,
            'prompt': prompt,
            'class1': class1,
            'class2': class2,
            'phrase': phrase,
            'question': question,
            'full_answer': full_answer,  # Keep the full answer for reference
            'answer': clean_answer       # Just "Yes" or "No"
        })
    
    return results


def main():
    parser = argparse.ArgumentParser(description="Process a specific phrase with LLaVa")
    parser.add_argument("phrase", type=str, help="The phrase to process (e.g., 'on top of', 'inside', etc.)")
    parser.add_argument("--examples", type=str, help="Path to examples CSV file for in-context learning")
    args = parser.parse_args()
    
    phrase = args.phrase
    
    # Load all metadata CSVs from all tensor directories
    print("Loading metadata from all directories...")
    all_metadata = []
    for tensor_dir in tensor_dirs:
        metadata_df = load_metadata_from_dir(tensor_dir)
        if not metadata_df.empty:
            all_metadata.append(metadata_df)
    
    # Find the phrase in metadata
    print(f"Searching for phrase: '{phrase}'")
    phrase_data, found_dirs = find_phrase_in_metadata(phrase, all_metadata)
    
    if phrase_data is None or phrase_data.empty:
        print(f"No data found for phrase: '{phrase}'")
        return
    
    print(f"Found {len(phrase_data)} entries for phrase '{phrase}' in directories:")
    for d in found_dirs:
        print(f"  - {d}")
    
    # Load examples if provided
    examples_df = None
    if args.examples:
        print(f"Loading examples from {args.examples}")
        examples_df = load_examples_csv(args.examples)
    print(examples_df)
    
    # Process the phrase data with or without in-context examples
    if examples_df is not None:
        results = process_phrase_data_with_incontext(phrase_data, found_dirs, phrase, examples_df, all_metadata)
    else:
        results = process_phrase_data(phrase_data, found_dirs, phrase)
    
    # Save results
    if results:
        os.makedirs("llava_outputs", exist_ok=True)
        output_df = pd.DataFrame(results)
        safe_phrase = phrase.replace(" ", "_")
        incontext_tag = "_incontext" if examples_df is not None else ""
        
        output_path = os.path.join("llava_outputs", f"{safe_phrase}{incontext_tag}_llava_output_{timestamp}.csv")
        output_df.to_csv(output_path, index=False)
        print(f"Saved {len(results)} results to {output_path}")
    else:
        print(f"No results generated for phrase: '{phrase}'")

if __name__ == "__main__":
    main()
