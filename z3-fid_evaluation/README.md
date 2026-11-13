# FID Evaluation for Custom SD-3.5

Evaluate image quality of Custom SD-3.5 (with CLIC text encoder) compared to Standard SD-3.5 using FID scores against COCO ground-truth.`

## Pipeline Steps

### 0. Download COCO Captions
- Downloads COCO 2017 validation annotations
- Samples 1,000 random captions
- Saves to `coco_1k_captions.csv`

### 1. Prepare COCO Ground-Truth (REQUIRED)
- Downloads COCO 2017 validation images (~1GB)
- Extracts the 1k images matching our captions
- Resizes to 512x512 and stacks into tensor
- Saves as single file: `coco_ground_truth.pt`

### 2. Generate Images
- Loads Standard SD-3.5 pipeline
- Generates 1k images using COCO captions
- Loads Custom SD-3.5 pipeline (with CLIC encoder)
- Generates 1k images using same captions
- Saves each model's images as single tensor file


**Each .pt file contains:**
```python
{
    'tensors': torch.FloatTensor([1000, 3, 512, 512]),  # All images
    'captions': list of 1000 strings,
    'image_ids': list of 1000 COCO IDs,
    'num_images': 1000,
    'height': 512,
    'width': 512
}
```

### 3. Calculate FID Scores
- Compares Standard SD-3.5 against COCO ground-truth (baseline)
- Compares Custom SD-3.5 against COCO ground-truth (evaluation)
- Loads full tensors from single files (much faster I/O)
- Saves all results to `fid_results.csv`

## Interpreting Results

FID measures the distance between generated and real image distributions (lower = better):

**What the scores mean:**
- Compare the two FID values: `Standard vs COCO` and `Custom vs COCO`
- **If scores are similar (difference < 2)**: Custom encoder preserves quality
- **If Custom FID < Standard FID**: Custom encoder improves quality!
- **If Custom FID > Standard FID**: Custom encoder degrades quality
