# z9-fid_new

COCO evaluation pipeline with a selectable FID backend.

## What changed

- Ground-truth images are copied as raw COCO JPGs instead of being resized with PIL.
- Turbo generation uses `guidance_scale = 0.0` by default.
- FID computation defaults to `pytorch-fid`, with optional `clean-fid` via a flag.

## Files

- `1_download_captions.py`: sample 5k unique COCO val captions.
- `2_coco_ground_truth.py`: copy matching raw COCO JPGs into `coco_ground_truth_images_raw/`.
- `3_generate_images.py`: generate model outputs with turbo-safe CFG.
- `4_fid.py`: compute FID and append results to `fid_results.csv`.

## Run order

```bash
python 1_download_captions.py
python 2_coco_ground_truth.py
python 3_generate_images.py --model all
python 4_fid.py                 # Uses pytorch-fid (resizes COCO images on-demand)
python 4_fid.py --use-clean-fid # Uses clean-fid (raw COCO images)
```

## FID Backend Details

- **pytorch-fid (default):** COCO images are automatically resized to 512×512 to match generation dimensions, then FID is computed.
- **clean-fid:** Raw COCO images are used directly (clean-fid handles resizing internally).

## Dependency

Install the optional Clean-FID backend before running with `--use-clean-fid`:

```bash
pip install clean-fid
```

`pytorch-fid` is the default backend and is already listed in the main project requirements.

## Note

The current caption sample is still 5k unique COCO val2017 images because that is the available unique-image pool in the existing workflow. If you want a 30k-image evaluation, the caption sampling source needs to be expanded first.
