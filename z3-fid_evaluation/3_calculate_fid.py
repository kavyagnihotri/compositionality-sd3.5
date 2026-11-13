"""
Calculate FID scores using torch-fidelity

Compares tensors:
  1. Standard SD-3.5 vs COCO ground-truth (baseline)
  2. Custom SD-3.5 vs COCO ground-truth (evaluation)
"""

import torch
import pandas as pd
from pathlib import Path
from datetime import datetime
from torch.utils.data import Dataset
from torch_fidelity import calculate_metrics


# Configuration
TENSOR_FILES = {
    'standard': "./generated_images/standard_sd35.pt",
    'custom': "./generated_images/custom_sd35.pt",
    'coco_gt': "./coco_ground_truth.pt",
}

OUTPUT_CSV = "./fid_results.csv"
BATCH_SIZE = 50
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


class TensorDataset(Dataset):
    """
    Dataset for pre-loaded tensors.
    torch-fidelity expects uint8 [0, 255] range, so we convert.
    """
    def __init__(self, tensors):
        """
        Args:
            tensors: torch.Tensor of shape [N, C, H, W] in range [0, 1]
        """
        # Convert to [0, 255] uint8 for torch-fidelity
        self.tensors = (tensors * 255).byte()
        print(f"  Loaded {len(self.tensors)} images, shape: {self.tensors.shape}")
    
    def __len__(self):
        return len(self.tensors)
    
    def __getitem__(self, idx):
        return self.tensors[idx]


def calculate_fid_from_tensors(tensor_file1, tensor_file2, name1, name2):
    """
    Calculate FID score between two tensor files using torch-fidelity.
    
    Args:
        tensor_file1: Path to first tensor file (.pt)
        tensor_file2: Path to second tensor file (.pt)
        name1: Name of first dataset
        name2: Name of second dataset
    
    Returns:
        FID score (float)
    """
    print(f"\nCalculating FID: {name1} vs {name2}")
    print(f"  File 1: {tensor_file1}")
    print(f"  File 2: {tensor_file2}")
    
    # Load tensor data
    data1 = torch.load(tensor_file1)
    data2 = torch.load(tensor_file2)
    
    # Extract tensors [N, C, H, W]
    tensors1 = data1['tensors']
    tensors2 = data2['tensors']
    
    # Create datasets
    dataset1 = TensorDataset(tensors1)
    dataset2 = TensorDataset(tensors2)
    
    if len(dataset1) == 0 or len(dataset2) == 0:
        print(f"  Error: Empty dataset(s)")
        return None
    
    # Calculate FID using torch-fidelity
    print(f"  Computing FID with batch_size={BATCH_SIZE}...")
    
    metrics = calculate_metrics(
        input1=dataset1,
        input2=dataset2,
        cuda=(DEVICE == "cuda"),
        batch_size=BATCH_SIZE,
        fid=True,
        verbose=False
    )
    
    fid_score = metrics['frechet_inception_distance']
    print(f"  FID Score: {fid_score:.4f}")
    
    return fid_score


def main():
    print("FID SCORE CALCULATION")
    print(f"Device: {DEVICE}")
    
    results = []
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    
    # Check if tensor files exist
    standard_file = Path(TENSOR_FILES['standard'])
    custom_file = Path(TENSOR_FILES['custom'])
    coco_gt_file = Path(TENSOR_FILES['coco_gt'])
    
    if not standard_file.exists():
        print(f"\nError: Standard SD-3.5 tensor not found at {standard_file}")
        return
    
    if not custom_file.exists():
        print(f"\nError: Custom SD-3.5 tensor not found at {custom_file}")
        return
    
    if not coco_gt_file.exists():
        print(f"\nError: COCO ground-truth not found at {coco_gt_file}")
        return
    
    # Load and verify tensor files
    standard_data = torch.load(standard_file)
    custom_data = torch.load(custom_file)
    coco_data = torch.load(coco_gt_file)
    
    standard_count = standard_data['num_images']
    custom_count = custom_data['num_images']
    coco_count = coco_data['num_images']
    
    print(f"  Standard SD-3.5: {standard_count} images")
    print(f"  Custom SD-3.5:   {custom_count} images")
    print(f"  COCO GT:         {coco_count} images")
    
    # ============ Comparison 1: Standard vs COCO ============
    print("COMPARISON 1: Standard SD-3.5 vs COCO Ground-Truth")
    
    fid_standard_coco = calculate_fid_from_tensors(
        standard_file,
        coco_gt_file,
        "Standard SD-3.5",
        "COCO Ground-Truth"
    )
    
    if fid_standard_coco is not None:
        results.append({
            'comparison': 'Standard_vs_COCO',
            'dataset_1': 'Standard SD-3.5',
            'dataset_2': 'COCO Ground-Truth',
            'fid_score': fid_standard_coco,
            'num_images': min(standard_count, coco_count),
            'timestamp': timestamp
        })
    
    # ============ Comparison 2: Custom vs COCO ============
    print("COMPARISON 2: Custom SD-3.5 vs COCO Ground-Truth")
    
    fid_custom_coco = calculate_fid_from_tensors(
        custom_file,
        coco_gt_file,
        "Custom SD-3.5",
        "COCO Ground-Truth"
    )
    
    if fid_custom_coco is not None:
        results.append({
            'comparison': 'Custom_vs_COCO',
            'dataset_1': 'Custom SD-3.5',
            'dataset_2': 'COCO Ground-Truth',
            'fid_score': fid_custom_coco,
            'num_images': min(custom_count, coco_count),
            'timestamp': timestamp
        })
    
    # ============ Save Results ============
    if results:
        df = pd.DataFrame(results)
        output_path = Path(OUTPUT_CSV)
        df.to_csv(output_path, index=False)
        
        print("FID RESULTS SUMMARY")
        print(df.to_string(index=False))
        print(f"\nSaved results to {output_path}")
    else:
        print("\nNo FID scores calculated")


if __name__ == "__main__":
    main()