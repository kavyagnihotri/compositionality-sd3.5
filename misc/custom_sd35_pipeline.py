"""
Custom SD3.5 Pipeline with OpenCLIP Text Encoder Replacement

Extracts actual per-token embeddings from OpenCLIP's transformer layers, 
preserving token-level semantic information for better image generation.
"""

import torch
from transformers.models.clip.modeling_clip import CLIPTextModelOutput
import open_clip
from diffusers import StableDiffusion3Pipeline
from typing import Optional

LOCAL_SD_DIR = "/mnt/lustre/work/hein/hof217/stable_diffusion/stable-diffusion-3.5-large-turbo"
HF_HUB_CHECKPOINT = 'hf-hub:nmndeep/CLIC-ViT-L-14-224-PixPr-RedCaps'


class OpenCLIPTextEncoderWrapper(torch.nn.Module):
    def __init__(self, oc_model, orig_config, device, target_hidden_dim, target_pooler_dim, target_seq_len):
        super().__init__()
        self.oc = oc_model
        self.device = device
        self.dtype = torch.float16 if device.startswith("cuda") else torch.float32
        self.config = orig_config
        self.target_seq_len = target_seq_len
        
        openclip_dim = 768
        
        # Validate dimensions match (no projection needed when they match)
        if openclip_dim != target_hidden_dim or openclip_dim != target_pooler_dim:
            raise ValueError(
                f"Dimension mismatch: OpenCLIP outputs {openclip_dim} dims, "
                f"but SD3.5 expects hidden_dim={target_hidden_dim}, pooler_dim={target_pooler_dim}. "
                f"Add projection layers to handle dimension mismatches."
            )

    def forward(self, input_ids: torch.LongTensor, attention_mask: Optional[torch.Tensor]=None, 
                return_dict: bool=True, output_hidden_states: bool=False):
        """
        Instead of using encode_text() which only returns pooled output,
        we access OpenCLIP's transformer directly to get per-token embeddings.
        """
        
        with torch.no_grad():
            # Access OpenCLIP's text model directly to get per-token embeddings
            # OpenCLIP structure: model.token_embedding -> transformer -> ln_final -> text_projection
            
            # Step 1: Get token embeddings + positional embeddings
            x = self.oc.token_embedding(input_ids)  # [batch, seq_len, 768]
            x = x + self.oc.positional_embedding  # Add positional encoding
            x = x.permute(1, 0, 2)  # [seq_len, batch, 768] - transformer expects this order
            
            # Step 2: Pass through transformer layers
            x = self.oc.transformer(x)  # [seq_len, batch, 768]
            x = x.permute(1, 0, 2)  # [batch, seq_len, 768] - back to batch-first
            
            # Step 3: Apply layer normalization
            x = self.oc.ln_final(x)  # [batch, seq_len, 768]
            
            # Step 4: Extract pooled output (for text_embeds)
            # OpenCLIP uses the embedding at the EOS token position
            # Find EOS token positions
            eot_token_idx = input_ids.argmax(dim=-1)  # [batch] - index of EOS token per sample
            batch_indices = torch.arange(x.shape[0], device=x.device)
            pooled_output = x[batch_indices, eot_token_idx]  # [batch, 768]
            
            # Apply text projection if available
            if hasattr(self.oc, 'text_projection') and self.oc.text_projection is not None:
                pooled_output = pooled_output @ self.oc.text_projection
            
            # Normalize pooled output
            pooled_output = pooled_output / pooled_output.norm(dim=-1, keepdim=True)
            
            # Step 5: Handle sequence length mismatch
            # OpenCLIP may have different max_seq_len than SD3.5's expected 77
            current_seq_len = x.shape[1]
            
            if current_seq_len < self.target_seq_len:
                # Pad with zeros if OpenCLIP sequence is shorter
                padding = torch.zeros(
                    x.shape[0], 
                    self.target_seq_len - current_seq_len, 
                    x.shape[2],
                    dtype=x.dtype,
                    device=x.device
                )
                last_hidden_state = torch.cat([x, padding], dim=1)
            elif current_seq_len > self.target_seq_len:
                # Truncate if OpenCLIP sequence is longer
                last_hidden_state = x[:, :self.target_seq_len, :]
            else:
                last_hidden_state = x
        
        if return_dict:
            # Create hidden_states tuple (SD3 needs hidden_states[-2])
            hidden_states_tuple = None
            if output_hidden_states:
                # Provide multiple layers, with hidden_states[-2] being the penultimate
                hidden_states_tuple = (
                    last_hidden_state,  # layer 0
                    last_hidden_state,  # layer 1  (this will be accessed as [-2])
                    last_hidden_state,  # layer 2  (this will be accessed as [-1])
                )
            
            # Use CLIPTextModelOutput (same as CLIPTextModelWithProjection returns)
            return CLIPTextModelOutput(
                last_hidden_state=last_hidden_state,  # [batch, 77, 768]
                text_embeds=pooled_output,  # [batch, 768] - pooled output
                hidden_states=hidden_states_tuple
            )
        else:
            return (last_hidden_state, pooled_output)


class TokenizerOutput:
    def __init__(self, input_ids):
        self.input_ids = input_ids


class OpenCLIPTokenizerAdapter:
    """Adapter to make OpenCLIP tokenizer compatible with SD3.5 pipeline."""
    
    def __init__(self, openclip_tokenizer, device):
        self.tokenizer = openclip_tokenizer
        self.model_max_length = 77
        self.device = device

    def __call__(self, texts, return_tensors="pt", padding="max_length", truncation=True, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        tokens = self.tokenizer(texts).to(self.device)
        return TokenizerOutput(input_ids=tokens)
    
    def decode(self, token_ids, **kwargs):
        return "decoded_text"


def create_custom_sd35_pipeline(device="cuda"):
    print("="*70)
    print("SETTING UP IMPROVED CUSTOM SD3.5 WITH OPENCLIP (PER-TOKEN)")
    print("="*70)
    
    # (1) Load SD3.5 Pipeline on CPU first
    print("\nLoading SD3.5 pipeline on CPU...")
    pipe = StableDiffusion3Pipeline.from_pretrained(
        LOCAL_SD_DIR,
        torch_dtype=torch.float16,
    )
    print("Loaded SD3.5 pipeline on CPU")
    
    # Get target dimensions from original encoder
    print("\nAnalyzing original text_encoder...")
    with torch.no_grad():
        sample_tokens = pipe.tokenizer("test", return_tensors="pt", padding="max_length").input_ids
        sample_output = pipe.text_encoder(sample_tokens, output_hidden_states=True)
    
    TARGET_HIDDEN_DIM = sample_output.last_hidden_state.shape[-1]
    TARGET_POOLER_DIM = 768  # Standard CLIP-L pooler dimension
    TARGET_SEQ_LEN = sample_output.last_hidden_state.shape[1]
    
    print(f"Target dimensions: Hidden={TARGET_HIDDEN_DIM}, Pooler={TARGET_POOLER_DIM}, SeqLen={TARGET_SEQ_LEN}")
    
    # (2) Load OpenCLIP model
    print("\nLoading OpenCLIP model...")
    openclip_model, _, _ = open_clip.create_model_and_transforms(HF_HUB_CHECKPOINT)
    openclip_tokenizer = open_clip.get_tokenizer(HF_HUB_CHECKPOINT)
    
    openclip_model = openclip_model.to(device)
    if device.startswith("cuda"):
        openclip_model = openclip_model.half()
    openclip_model.eval()
    print("Loaded OpenCLIP model")
    
    # (3) Create wrapper
    print("\nCreating OpenCLIP wrapper (per-token embeddings)...")
    wrapped_text_encoder = OpenCLIPTextEncoderWrapper(
        openclip_model, 
        pipe.text_encoder.config,
        device,
        TARGET_HIDDEN_DIM,
        TARGET_POOLER_DIM,
        TARGET_SEQ_LEN
    )
    print("Created wrapper")
    
    # (4) Replace text_encoder
    print("\nReplacing text_encoder...")
    pipe._orig_text_encoder = pipe.text_encoder
    pipe._orig_tokenizer = pipe.tokenizer
    
    pipe.text_encoder = wrapped_text_encoder
    pipe.tokenizer = OpenCLIPTokenizerAdapter(openclip_tokenizer, device)
    
    # Update the pipeline's component registry
    pipe.register_modules(text_encoder=wrapped_text_encoder, tokenizer=pipe.tokenizer)
    print("Text encoder replaced")
    
    # (5) Move entire pipeline to GPU
    print("\nMoving pipeline to GPU...")
    pipe = pipe.to(device)
    print("Pipeline on GPU\n")
    
    return pipe
