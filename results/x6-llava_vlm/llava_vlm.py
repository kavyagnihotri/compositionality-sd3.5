import os
import torch
import pandas as pd
from PIL import Image
from pathlib import Path
import argparse
from huggingface_hub import InferenceClient
from tqdm import tqdm
import datetime
import dotenv
import io
import base64

dotenv.load_dotenv()

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


def image_to_base64(image):
    """Convert PIL Image to base64 string"""
    buffered = io.BytesIO()
    image.save(buffered, format="PNG")
    img_str = base64.b64encode(buffered.getvalue()).decode()
    return f"data:image/png;base64,{img_str}"


def create_qwen_message(image, class1, phrase, class2):
    """Create a message for the Qwen model"""
    question = f"Is a {class1} {phrase} a {class2}? Answer in Yes or No"
    
    # Convert image to base64
    image_url = image_to_base64(image)
    
    messages = [
        {
            "role": "user",
            "content": [
                {
                    "type": "text",
                    "text": question
                },
                {
                    "type": "image_url",
                    "image_url": {
                        "url": image_url
                    }
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
    
    # Initialize Qwen client
    print("Initializing Qwen client...")
    client = InferenceClient(
        provider="auto",
        api_key=os.getenv("HF_TOKEN"),
    )
    print("Client initialized")
    
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
            
        # Load the image from tensor
        image = load_tensor_as_image(tensor_path, tensor_index)
        if image is None:
            continue
            
        # Create message and get response
        messages, question = create_qwen_message(image, class1, phrase, class2)
        
        try:
            # Process with Qwen
            completion = client.chat.completions.create(
                model="Qwen/Qwen3-VL-235B-A22B-Instruct",
                messages=messages,
            )
            
            full_answer = completion.choices[0].message.content.strip()
            
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