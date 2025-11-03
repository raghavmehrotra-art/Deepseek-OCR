"""CLIP-like Stage: Global attention with vision-language knowledge.

Uses CLIP vision encoder layers (without initial patch embedding) to process
compressed tokens with dense global attention.

Paper reference: Section 3.2.1 - Stage 2 uses CLIP-large with first patch
embedding layer removed, accepting tokens from compressor instead.
"""
import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import CLIPVisionModel, CLIPVisionConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class CLIPLikeStage(nn.Module):
    """Second stage of DeepEncoder: global attention with CLIP.
    
    Uses CLIP vision transformer layers (excluding patch embedding)
    to process compressed tokens with global attention.
    
    Args:
        in_dim: input dimension from compressor
        clip_model_name: HF CLIP model name
        use_pretrained: load pretrained CLIP weights
        num_layers: number of CLIP layers to use (None = all)
    """
    def __init__(
        self,
        in_dim: int = 1024,
        clip_model_name: str = "openai/clip-vit-base-patch32",
        use_pretrained: bool = True,
        num_layers: Optional[int] = None
    ):
        super().__init__()
        self.in_dim = in_dim
        
        if HF_AVAILABLE:
            # Load CLIP vision model
            if use_pretrained:
                clip_full = CLIPVisionModel.from_pretrained(clip_model_name)
            else:
                config = CLIPVisionConfig.from_pretrained(clip_model_name)
                clip_full = CLIPVisionModel(config)
            
            self.hidden_size = clip_full.config.hidden_size
            
            # Input projection to CLIP hidden size
            self.input_proj = nn.Linear(in_dim, self.hidden_size)
            
            # Extract encoder layers (skip patch embedding)
            self.encoder = clip_full.vision_model.encoder
            if num_layers is not None:
                # Use only first N layers
                self.encoder.layers = self.encoder.layers[:num_layers]
            
            # Layer norm and projection
            self.ln = clip_full.vision_model.post_layernorm
            self.proj = nn.Linear(self.hidden_size, in_dim)
        else:
            print("Warning: transformers not available, using simple projection")
            self.hidden_size = in_dim
            self.input_proj = nn.Identity()
            self.encoder = nn.TransformerEncoder(
                nn.TransformerEncoderLayer(
                    d_model=in_dim,
                    nhead=8,
                    dim_feedforward=in_dim * 4,
                    batch_first=True
                ),
                num_layers=6
            )
            self.ln = nn.LayerNorm(in_dim)
            self.proj = nn.Identity()
    
    def forward(self, tokens: torch.Tensor) -> torch.Tensor:
        """Forward pass with global attention.
        
        Args:
            tokens: [B, N, in_dim] compressed tokens from compressor
            
        Returns:
            output: [B, N, in_dim] globally-attended features
        """
        # Project to CLIP hidden size
        x = self.input_proj(tokens)  # [B, N, hidden_size]
        
        # Apply CLIP encoder layers
        if HF_AVAILABLE:
            x = self.encoder(x)[0]  # [B, N, hidden_size]
        else:
            x = self.encoder(x)  # [B, N, hidden_size]
        
        # Layer norm and project back
        x = self.ln(x)
        x = self.proj(x)  # [B, N, in_dim]
        
        return x
