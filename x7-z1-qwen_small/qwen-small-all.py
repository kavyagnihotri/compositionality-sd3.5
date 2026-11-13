import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse
from tqdm import tqdm
import datetime
import io
import base64
from transformers import Qwen3VLMoeForConditionalGeneration, AutoProcessor

model = Qwen3VLMoeForConditionalGeneration.from_pretrained(
    "Qwen/Qwen3-VL-30B-A3B-Instruct",
    dtype="auto",
    device_map="auto",
    low_cpu_mem_usage=True,
)
processor = AutoProcessor.from_pretrained("Qwen/Qwen3-VL-30B-A3B-Instruct")

# Directories containing tensor data
tensor_dirs = ["sd35_output_behind_on_front", "sd35_output_below_over_under",
               "sd35_output_between_inside_outside", "sd35_output_holding_touching_not", 
               "sd35_output_left_right_above", "sd35_output_top"]
# tensor_dirs = ["sd35_output_20251103_131309"]
base_path = "../sept_w4" 
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


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_qwen_message(image, class1, phrase, class2):
    """Create a message for the Qwen model"""
    question = f"Does this figure show 'a {class1} {phrase} a {class2}'? Please answer yes, no."
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image",
                    "image": image
                }
            ]
        }
    ]
    return messages, question


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
    
    # model and processor are initialized at module import time
    model.eval()
    # Determine target device for tensors (inputs should be moved to a CUDA device if available)
    target_device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    
    # Process each entry for the phrase
    for _, row in tqdm(phrase_data.iterrows(), total=len(phrase_data), desc=f"Processing '{phrase}'"):
        tensor_file = row['tensor_file']
        tensor_index = int(row['tensor_index'])
        class1 = row['class1']
        class2 = row['class2']
        prompt = row.get('prompt', '')
        
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
            
        # Load the image from tensor
        image = load_tensor_as_image(tensor_path, tensor_index)
        if image is None:
            continue
            
        # Create message and get response
        messages, question = create_qwen_message(image, class1, phrase, class2)
        
        try:
            # Prepare inputs via the processor (tokenize and convert to tensors)
            inputs = processor.apply_chat_template(
                messages,
                tokenize=True,
                add_generation_prompt=True,
                return_dict=True,
                return_tensors="pt",
            )

            # Move any tensor inputs to target device (avoid input/device mismatch warnings)
            for k, v in list(inputs.items()):
                if isinstance(v, torch.Tensor):
                    inputs[k] = v.to(target_device)

            # Generate using the model
            generated_ids = model.generate(**inputs, max_new_tokens=128)

            # Trim the generated portion and decode to text
            generated_ids_trimmed = [
                out_ids[len(in_ids):] for in_ids, out_ids in zip(inputs["input_ids"], generated_ids)
            ]
            full_answer = processor.batch_decode(
                generated_ids_trimmed, skip_special_tokens=True, clean_up_tokenization_spaces=False
            )
            # full_answer is a list; take first element
            full_answer_text = full_answer[0].strip() if isinstance(full_answer, (list, tuple)) else str(full_answer).strip()
            
            # Extract just "Yes" or "No" from the answer
            if "yes" in full_answer_text.lower():
                clean_answer = "YES"
            elif "no" in full_answer_text.lower():
                clean_answer = "NO"
            else:
                clean_answer = "Unclear"
            
            # Store result
            results.append({
                'tensor_file': tensor_file,
                'tensor_index': tensor_index,
                'class1': class1,
                'class2': class2,
                'phrase': phrase,
                'question': question,
                'full_answer': full_answer_text,
                'answer': clean_answer
            })
        except Exception as e:
            print(f"Error processing image {tensor_file}[{tensor_index}]: {str(e)}")
            continue
    
    return results

def main():
    parser = argparse.ArgumentParser(description="Process a specific phrase with Qwen")
    parser.add_argument("phrase", type=str, help="The phrase to process (e.g., 'on top of', 'inside', etc.)")
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
    
    # Process the phrase data
    results = process_phrase_data(phrase_data, found_dirs, phrase)
    
    # Save results
    if results:
        os.makedirs("qwen_outputs", exist_ok=True)
        output_df = pd.DataFrame(results)
        safe_phrase = phrase.replace(" ", "_")

        output_path = os.path.join("qwen_outputs", f"{safe_phrase}_qwen_output_{timestamp}.csv")
        output_df.to_csv(output_path, index=False)
        print(f"Saved {len(results)} results to {output_path}")
    else:
        print(f"No results generated for phrase: '{phrase}'")

if __name__ == "__main__":
    main()