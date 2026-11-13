import torch
from collections import OrderedDict

from transformers import CLIPVisionModel, CLIPVisionConfig
import open_clip
import os

def convert_open_clip_to_transformers_vision_model(open_clip_checkpoint_path, save_path):
    """
    Converts an open_clip ViT-L-14-336 vision model checkpoint to a transformers CLIPVisionModel.

    Args:
        open_clip_checkpoint_path (str): Path to your open_clip model checkpoint (e.g., .pt file).
        save_path (str): Directory to save the converted transformers model.
    """
    print(f"Loading open_clip checkpoint from: {open_clip_checkpoint_path}")

    # --- Step 1: Load open_clip model (adjusted for ViT-L-14-336) ---
    try:
        loaded_open_clip_model = torch.load(open_clip_checkpoint_path, map_location="cpu")
        
        open_clip_vision_state_dict = {}
        if isinstance(loaded_open_clip_model, dict):
            if 'state_dict' in loaded_open_clip_model:
                # If it's a full open_clip checkpoint with 'state_dict' key
                open_clip_full_state_dict = loaded_open_clip_model['state_dict']
            else:
                # If it's just the raw state_dict
                open_clip_full_state_dict = loaded_open_clip_model

            # Filter open_clip state_dict to only include vision relevant keys
            # Common prefixes for vision encoder in open_clip
            for k, v in open_clip_full_state_dict.items():
                if k.startswith('visual.'):
                    open_clip_vision_state_dict[k.replace('visual.', '')] = v
                elif k.startswith('vision_model.'):
                    open_clip_vision_state_dict[k.replace('vision_model.', '')] = v
                elif k.startswith('module.visual.'): # For DDP wrapped models
                    open_clip_vision_state_dict[k.replace('module.visual.', '')] = v
                elif k.startswith('transformer.visual.'): # Another possible prefix
                    open_clip_vision_state_dict[k.replace('transformer.visual.', '')] = v
                # If your checkpoint is already only the vision encoder, no prefix might be needed
                elif not any(prefix in k for prefix in ['text.', 'logit_scale', 'token_embedding', 'positional_embedding', 'ln_final', 'transformer.text_model']):
                    # This is a heuristic: if no common text encoder prefixes, assume it's part of vision
                    # Be cautious with this. It's best if the vision part is explicitly prefixed.
                    if 'conv1.weight' in k or 'transformer.resblocks.' in k: # Look for distinct vision keys
                         open_clip_vision_state_dict[k] = v


            if not open_clip_vision_state_dict:
                print("Warning: No 'visual.' or 'vision_model.' or common vision prefixes found. Assuming the loaded state_dict is directly the vision model.")
                open_clip_vision_state_dict = open_clip_full_state_dict
                # This could mean your checkpoint is already just the vision model,
                # or you need to adjust the prefix filtering.
        else: # Assuming it's a full open_clip model object that was saved
            print("Loaded open_clip checkpoint as a full model object. Extracting visual part.")
            if hasattr(loaded_open_clip_model, 'visual'):
                open_clip_vision_state_dict = loaded_open_clip_model.visual.state_dict()
            else:
                raise ValueError("Could not find 'visual' attribute in the loaded open_clip model. "
                                 "Please ensure the checkpoint contains a vision encoder or adjust loading logic.")
    except Exception as e:
        print(f"Error loading open_clip checkpoint: {e}")
        print("Please ensure your open_clip checkpoint is valid and you're providing the correct path.")
        return

    # --- Step 2: Create transformers CLIPVisionConfig for ViT-L-14-336 ---
    # These parameters are specific to ViT-L-14-336
    config = CLIPVisionConfig(
        hidden_size=1024,          # Transformer width for ViT-L
        intermediate_size=4096,    # hidden_size * mlp_ratio (1024 * 4)
        num_hidden_layers=24,      # Number of transformer layers for ViT-L
        num_attention_heads=16,    # Number of attention heads for ViT-L
        image_size=336,            # Input image resolution for ViT-L-14-336
        patch_size=14,             # Patch size for ViT-L-14
        projection_dim=768,        # The dimension of the final projected features
                                   # In transformers CLIP, this is 768 for ViT-L
        layer_norm_eps=1e-05,
        initializer_range=0.02,
        attention_dropout=0.0,
        num_channels=3,
        hidden_act="quick_gelu", # Original CLIP used QuickGELU, transformers defaults to GELU.
                                 # If open_clip used QuickGELU and you want exact match, set this.
                                 # open_clip often uses QuickGELU for OpenAI pretrained models.
    )

    print("Created transformers CLIPVisionConfig:")
    print(config)

    # --- Step 3: Instantiate transformers CLIPVisionModel ---
    transformers_model = CLIPVisionModel(config)
    transformers_state_dict = transformers_model.state_dict()

    # --- Step 4: Map State Dictionary Keys ---
    new_state_dict = {}
    for k, v in open_clip_vision_state_dict.items():
        # Patch embeddings
        if k == 'conv1.weight':
            new_state_dict['embeddings.patch_embedding.weight'] = v
        # NOTE: open_clip's conv1 typically does NOT have bias. If yours does, handle it.
        elif k == 'conv1.bias':
            pass 
        elif k == 'class_embedding':
            new_state_dict['embeddings.class_embedding'] = v
        elif k == 'positional_embedding':
            new_state_dict['embeddings.position_embedding.weight'] = v

        # Transformer encoder layers
        elif k.startswith('transformer.resblocks.'):
            layer_num = int(k.split('.')[2])
            # Attention
            if '.attn.in_proj_weight' in k: # Fused QKV
                q_len = v.shape[0] // 3
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.q_proj.weight'] = v[:q_len, :]
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.k_proj.weight'] = v[q_len:2*q_len, :]
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.v_proj.weight'] = v[2*q_len:, :]
                
            elif '.attn.in_proj_bias' in k: # Fused QKV bias
                q_len = v.shape[0] // 3
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.q_proj.bias'] = v[:q_len]
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.k_proj.bias'] = v[q_len:2*q_len]
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.v_proj.bias'] = v[2*q_len:]

            elif '.attn.out_proj.weight' in k:
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.out_proj.weight'] = v
            elif '.attn.out_proj.bias' in k:
                new_state_dict[f'encoder.layers.{layer_num}.self_attn.out_proj.bias'] = v

            # Layer norms
            elif '.ln_1.weight' in k:
                new_state_dict[f'encoder.layers.{layer_num}.layer_norm1.weight'] = v
            elif '.ln_1.bias' in k:
                new_state_dict[f'encoder.layers.{layer_num}.layer_norm1.bias'] = v
            elif '.ln_2.weight' in k:
                new_state_dict[f'encoder.layers.{layer_num}.layer_norm2.weight'] = v
            elif '.ln_2.bias' in k:
                new_state_dict[f'encoder.layers.{layer_num}.layer_norm2.bias'] = v

            # MLP (Feed-forward)
            elif '.mlp.c_fc.weight' in k:
                new_state_dict[f'encoder.layers.{layer_num}.mlp.fc1.weight'] = v
            elif '.mlp.c_fc.bias' in k:
                new_state_dict[f'encoder.layers.{layer_num}.mlp.fc1.bias'] = v
            elif '.mlp.c_proj.weight' in k:
                new_state_dict[f'encoder.layers.{layer_num}.mlp.fc2.weight'] = v
            elif '.mlp.c_proj.bias' in k:
                new_state_dict[f'encoder.layers.{layer_num}.mlp.fc2.bias'] = v

        # Final layer norm
        elif k == 'ln_post.weight':
            new_state_dict['encoder.final_layer_norm.weight'] = v
        elif k == 'ln_post.bias':
            new_state_dict['encoder.final_layer_norm.bias'] = v

        # Vision transformer projection (if present)
        # For ViT-L-14, the 'proj' layer in open_clip maps to 'visual_projection' in transformers CLIPVisionModel.
        # Ensure that `config.projection_dim` is set to the correct value (768 for ViT-L).
        elif k == 'proj.weight': # Weights of the final projection layer
            # open_clip's `proj` (output of visual) is (hidden_size, projection_dim) i.e., (1024, 768)
            # transformers `visual_projection` expects (projection_dim, hidden_size) for weight if transposed.
            # Usually, open_clip's `proj` weight needs to be transposed for transformers.
            new_state_dict['visual_projection.weight'] = v.T # Transpose (768, 1024)
        elif k == 'proj.bias': # Bias of the final projection layer
            # transformers `visual_projection` has bias
            new_state_dict['visual_projection.bias'] = v
            
        elif k == 'text_projection': # Should not be in a vision-only model state dict
            print(f"Warning: Found 'text_projection' in open_clip vision state_dict. Skipping as this is for text encoder.")
            pass
        elif k == 'logit_scale': # Should not be in a vision-only model state dict
            print(f"Warning: Found 'logit_scale' in open_clip vision state_dict. Skipping as this is for full CLIP model.")
            pass
        else:
            # Catch any unmapped keys for debugging
            if k not in new_state_dict:
                print(f"Warning: Unmapped key '{k}' found in open_clip state_dict. Skipping.")

    # Validate if all expected transformers keys are present and mapped
    missing_keys = set(transformers_state_dict.keys()) - set(new_state_dict.keys())
    if missing_keys:
        print(f"Warning: The following keys are missing in the converted state_dict for transformers model: {missing_keys}")
        print("This might be expected if your open_clip model's architecture differs or if some components (e.g., visual_projection) are optional.")
    
    # Check for unexpected keys in the new_state_dict
    extra_keys = set(new_state_dict.keys()) - set(transformers_state_dict.keys())
    if extra_keys:
        print(f"Warning: The following keys in the converted state_dict are not expected by transformers model: {extra_keys}")

    # --- Step 5: Load Mapped State Dictionary ---
    # It's good practice to ensure the shapes match before loading.
    for k_new, v_new in new_state_dict.items():
        if k_new in transformers_state_dict:
            if transformers_state_dict[k_new].shape != v_new.shape:
                print(f"Shape mismatch for {k_new}: transformers {transformers_state_dict[k_new].shape} vs open_clip {v_new.shape}")
                # A common mismatch is transposed linear layer weights, specifically for 'proj'
                if k_new == 'visual_projection.weight' and v_new.shape[::-1] == transformers_state_dict[k_new].shape:
                    print(f"Transposing {k_new} to match shape.")
                    new_state_dict[k_new] = v_new.T
                else:
                    raise ValueError(f"Shape mismatch for {k_new}: transformers {transformers_state_dict[k_new].shape} vs open_clip {v_new.shape}. Cannot auto-fix.")
        else:
            print(f"Warning: Key {k_new} from mapped open_clip state_dict not found in transformers model. This key will be ignored.")


    try:
        transformers_model.load_state_dict(new_state_dict, strict=True)
        print("Successfully loaded state_dict into transformers CLIPVisionModel with strict=True.")
    except RuntimeError as e:
        print(f"Error loading state_dict into transformers CLIPVisionModel: {e}")
        print("This often indicates a mismatch in keys or shapes. Review the mapping and missing/unexpected keys.")
        print("Attempting to load with strict=False to see what loads (for debugging).")
        transformers_model.load_state_dict(new_state_dict, strict=False)
        print("Loaded with strict=False. Please inspect the model for partially loaded weights.")

    # --- Step 6: Save transformers Model ---
    os.makedirs(save_path, exist_ok=True)
    transformers_model.save_pretrained(save_path)
    config.save_pretrained(save_path) # Save the config as well
    print(f"Converted transformers CLIPVisionModel saved to: {save_path}")

    print("\n--- Verification ---")
    try:
        # Verify by loading the saved model
        reloaded_model = CLIPVisionModel.from_pretrained(save_path)
        print("Successfully reloaded the converted model using CLIPVisionModel.from_pretrained.")
        print(reloaded_model)
    except Exception as e:
        print(f"Error verifying the saved model: {e}")

# --- How to use the function ---
if __name__ == "__main__":
    # --- IMPORTANT ---
    # 1. Download an open_clip ViT-L-14-336 checkpoint.
    #    You can often find these on Hugging Face (e.g., under 'openai/clip-vit-large-patch14-336')
    #    or directly from the open_clip GitHub releases/models.
    #    Example: If downloading 'open_clip_pytorch_model.bin' from Hugging Face:
    #    `wget https://huggingface.co/openai/clip-vit-large-patch14-336/resolve/main/open_clip_pytorch_model.bin -O clip_vit_large_patch14_336.bin`
    #    
    # 2. Set the `open_clip_checkpoint_path` to your downloaded file.
    # 3. Ensure your checkpoint truly contains the vision encoder's weights.
    #    If it's a full CLIP model, the code attempts to extract the 'visual.' prefixed keys.
    #    If your checkpoint names keys differently, you'll need to adjust the filtering logic.

    # Example: If you have a real ViT-L-14-336 checkpoint
    # For demonstration, I'll create a dummy one, but you should use your actual file.
    
    print("Creating a dummy open_clip ViT-L-14-336 vision model state_dict for demonstration...")
    # This dummy class mirrors the expected structure of open_clip's ViT-L-14-336 vision part
    # class DummyOpenCLIPVisionTransformerL(torch.nn.Module):
    #     def __init__(self):
    #         super().__init__()
    #         # ViT-L-14-336 specific parameters
    #         hidden_size = 1024
    #         patch_size = 14
    #         image_size = 336
    #         mlp_ratio = 4
    #         num_layers = 24
            
    #         self.conv1 = torch.nn.Conv2d(3, hidden_size, kernel_size=patch_size, stride=patch_size, bias=False)
    #         self.class_embedding = torch.nn.Parameter(torch.randn(1, 1, hidden_size))
    #         # Number of patches + 1 for class token
    #         num_patches = (image_size // patch_size) * (image_size // patch_size)
    #         self.positional_embedding = torch.nn.Parameter(torch.randn(1, num_patches + 1, hidden_size))
    #         self.ln_post = torch.nn.LayerNorm(hidden_size)
    #         self.proj = torch.nn.Parameter(torch.randn(hidden_size, 768)) # Output projection to 768

    #         self.transformer = torch.nn.ModuleDict()
    #         self.transformer.resblocks = torch.nn.ModuleList()
    #         for i in range(num_layers):
    #             block = torch.nn.ModuleDict()
    #             block.ln_1 = torch.nn.LayerNorm(hidden_size)
    #             block.attn = torch.nn.ModuleDict()
    #             block.attn.in_proj_weight = torch.nn.Parameter(torch.randn(3 * hidden_size, hidden_size))
    #             block.attn.in_proj_bias = torch.nn.Parameter(torch.randn(3 * hidden_size))
    #             block.attn.out_proj = torch.nn.Linear(hidden_size, hidden_size)
                
    #             block.ln_2 = torch.nn.LayerNorm(hidden_size)
    #             block.mlp = torch.nn.ModuleDict()
    #             block.mlp.c_fc = torch.nn.Linear(hidden_size, hidden_size * mlp_ratio)
    #             block.mlp.c_proj = torch.nn.Linear(hidden_size * mlp_ratio, hidden_size)
    #             self.transformer.resblocks.append(block)

    #     def forward(self, x):
    #         pass # Not implementing full forward for dummy

    # dummy_open_clip_visual = DummyOpenCLIPVisionTransformerL()
    # # For open_clip's full model, the visual encoder is usually under 'visual' attribute.
    # # So we simulate that by putting the dummy state_dict under 'visual.' prefix.
    # # If your actual checkpoint saves just the visual part without 'visual.',
    # # you'd just save dummy_open_clip_visual.state_dict() directly.
    # dummy_open_clip_state_dict = {f'visual.{k}': v for k, v in dummy_open_clip_visual.state_dict().items()}

    # dummy_open_clip_checkpoint_path = "./checkpoints/dummy_open_clip_vit_l_14_336_vision_checkpoint.pt"
    # torch.save(dummy_open_clip_state_dict, dummy_open_clip_checkpoint_path)
    # print(f"Dummy open_clip ViT-L-14-336 vision checkpoint created at: {dummy_open_clip_checkpoint_path}")

    # # Set your desired save path for the transformers model
    transformers_model_save_path = "./checkpoints/converted_clic_vision_model_vit_l_14_336-updated"
    ckpt = torch.load("/mnt/nsingh/clic-rcaps-336-l-14.pt", map_location='cpu')  # ['model']

    state_dict = OrderedDict()
    try:
        for k, v in ckpt['state_dict'].items():
            state_dict[k.replace('module.model.', '')] = v
    except:
        for k, v in ckpt.items():
            state_dict[k.replace('module.model.', '')] = v
    state_dict = load_my_state_dict(model, state_dict)
    torch.save(state_dict, "./checkpoint/fixed_rcaps-l-14-clic.pt")
    # try:
    #     # model = load_checkpoint(model, args.load_pretrained_clip)
    #     model.load_state_dict(state_dict, strict=True)
    #     print("Loaded from ckpt")
    # except:
    #     print("Failed a bit")


    convert_open_clip_to_transformers_vision_model(
        open_clip_checkpoint_path="./checkpoint/fixed_rcaps-l-14-clic.pt",
        save_path=transformers_model_save_path
        ,config=config
    )
    
    # # Clean up dummy file (optional)
    # if os.path.exists(dummy_open_clip_checkpoint_path):
    #     os.remove(dummy_open_clip_checkpoint_path)
    #     print(f"Removed dummy checkpoint file: {dummy_open_clip_checkpoint_path}")
    # Load the model from your local save path
    model = CLIPVisionModel.from_pretrained("openai/clip-vit-large-patch14-336")
    config = CLIPVisionConfig.from_pretrained("openai/clip-vit-large-patch14-336") # It's good to load the config too, though push_to_hub might save it again.

    # Define your repository ID on Hugging Face Hub
    # This will be "your_username/your-model-name" or "your_organization/your-model-name"
    # Example: "your_username/clip-vit-l-14-336-vision"
    repo_id = "nmndeep/clic-vit-l-14-336-redcaps-qg" # <--- IMPORTANT: Change this!

    # Push to Hugging Face Hub
    # The .push_to_hub() method handles uploading the model weights, config, and other files.
    # It will also create the repository on the Hub if it doesn't exist.
    try:
        model.push_to_hub(
            repo_id,
            # You can add a commit message
            commit_message="Initial release of converted ViT-L-14-336 CLIP Vision Model from OpenCLIP",
            # Set to True if you want a private repository
            private=False,
            # You can add a model card to describe your model on the Hub
            # model_card="A Vision Transformer Large model (L/14) trained on 336x336 images, converted from an OpenCLIP checkpoint.",
            # If you need to specify a token directly (e.g., in Colab without CLI login)
            # token=os.environ.get("HF_TOKEN")
        )
        print(f"Model successfully pushed to https://huggingface.co/{repo_id}")
    except Exception as e:
        print(f"Error pushing model to Hugging Face Hub: {e}")
        print("Please ensure you are logged in, have write access, and the repo_id is correct.")


    # Alternatively, you can save locally and then push
    # This is what the `convert_open_clip_to_transformers_vision_model` function does,
    # and then you can call push_to_hub on the `transformers_model` instance directly after conversion.
    #
    # model.save_pretrained(transformers_model_save_path) # Already done by conversion script
    # model.push_to_hub(repo_id) # Call this after the save_pretrained in the conversion script
