"""DeepEncoder: Core vision encoder for DeepSeek-OCR.

Combines SAM-like window attention, convolutional compression, and CLIP-like
global attention to efficiently process high-resolution images into compact
latent vision tokens.

Paper reference: Section 3.2 - DeepEncoder is ~380M parameters, combining
80M SAM-base and 300M CLIP-large in series with 16x compression.
"""
import torch
import torch.nn as nn
from typing import Optional
from .sam_stage import SAMLikeStage
from .compressor import ConvCompressor
from .clip_stage import CLIPLikeStage


class DeepEncoder(nn.Module):
    """DeepEncoder: Two-stage vision encoder with compression.
    
    Architecture:
        Input Image [B, 3, H, W]
        ↓
        Stage 1: SAM-like (window attention) → [B, N, embed_dim]
        ↓
        Compressor (16x reduction) → [B, N/16, clip_dim]
        ↓
        Stage 2: CLIP-like (global attention) → [B, N/16, clip_dim]
    
    Args:
        embed_dim: embedding dimension for Stage 1
        clip_dim: embedding dimension for Stage 2 and output
        sam_model: timm model name for SAM-like stage
        clip_model: HF CLIP model name
        use_pretrained: load pretrained weights
    """
    def __init__(
        self,
        embed_dim: int = 256,
        clip_dim: int = 1024,
        sam_model: str = "swin_tiny_patch4_window7_224",
        clip_model: str = "openai/clip-vit-base-patch32",
        use_pretrained: bool = True
    ):
        super().__init__()
        self.embed_dim = embed_dim
        self.clip_dim = clip_dim
        
        # Stage 1: Window attention (SAM-like)
        self.stage1 = SAMLikeStage(
            model_name=sam_model,
            embed_dim=embed_dim,
            pretrained=use_pretrained
        )
        
        # Compressor: 16x token reduction
        self.compressor = ConvCompressor(
            in_ch=embed_dim,
            out_ch=clip_dim
        )
        
        # Stage 2: Global attention (CLIP-like)
        self.stage2 = CLIPLikeStage(
            in_dim=clip_dim,
            clip_model_name=clip_model,
            use_pretrained=use_pretrained
        )
    
    @torch.no_grad()
    def estimate_tokens(self, h: int, w: int, mode: str = "base") -> int:
        """Estimate number of output tokens for given input size.
        
        Args:
            h, w: input height and width
            mode: resolution mode (tiny/small/base/large)
            
        Returns:
            estimated token count
        """
        # After Stage 1 (patch_size=16): (h/16) * (w/16)
        # After Compressor (16x reduction): ((h/16) * (w/16)) / 16 = (h*w) / 4096
        return (h * w) // 4096
    
    def forward(
        self,
        images: torch.Tensor,
        return_intermediate: bool = False
    ) -> torch.Tensor:
        """Forward pass through DeepEncoder.
        
        Args:
            images: [B, 3, H, W] input images
            return_intermediate: if True, return all intermediate features
            
        Returns:
            tokens: [B, N, clip_dim] compressed vision tokens
            (optional) dict with intermediate features
        """
        # Stage 1: Window attention
        stage1_tokens, hw1 = self.stage1(images)
        
        # Compressor: 16x reduction
        compressed_tokens, hw2 = self.compressor(stage1_tokens, hw1)
        
        # Stage 2: Global attention
        output_tokens = self.stage2(compressed_tokens)
        
        if return_intermediate:
            return output_tokens, {
                "stage1_tokens": stage1_tokens,
                "stage1_shape": hw1,
                "compressed_tokens": compressed_tokens,
                "compressed_shape": hw2
            }
        
        return output_tokens
