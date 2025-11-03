"""SAM-like Stage: Window-attention dominated vision encoder.

Uses a Swin Transformer or similar window-attention architecture to process
high-resolution images efficiently. Outputs patch tokens and grid dimensions.

Paper reference: Section 3.2.1 - Stage 1 of DeepEncoder uses SAM-base with
patch size 16 and window attention for low activation at high resolutions.
"""
import torch
import torch.nn as nn
try:
    import timm
    TIMM_AVAILABLE = True
except ImportError:
    TIMM_AVAILABLE = False


class SAMLikeStage(nn.Module):
    """First stage of DeepEncoder: window-attention based feature extraction.
    
    Args:
        model_name: timm model name (e.g., 'swin_base_patch4_window7_224')
        embed_dim: output embedding dimension
        pretrained: load pretrained weights
        patch_size: patch size for tokenization (default 16 to match paper)
    """
    def __init__(
        self,
        model_name: str = "swin_tiny_patch4_window7_224",
        embed_dim: int = 256,
        pretrained: bool = True,
        patch_size: int = 16
    ):
        super().__init__()
        self.patch_size = patch_size
        self.embed_dim = embed_dim
        
        if TIMM_AVAILABLE:
            # Use Swin Transformer as SAM-like window attention backbone
            self.backbone = timm.create_model(
                model_name,
                pretrained=pretrained,
                features_only=True,
                out_indices=[1]  # Get intermediate features
            )
            # Get the feature dimension from the backbone
            with torch.no_grad():
                dummy = torch.randn(1, 3, 224, 224)
                feats = self.backbone(dummy)
                backbone_dim = feats[0].shape[1]
            
            # Project to target embedding dimension
            self.proj = nn.Linear(backbone_dim, embed_dim)
        else:
            # Fallback: simple conv patch embedding
            print("Warning: timm not available, using simple conv backbone")
            self.backbone = None
            self.proj = nn.Conv2d(3, embed_dim, kernel_size=patch_size, stride=patch_size)
    
    def forward(self, x: torch.Tensor) -> tuple[torch.Tensor, tuple[int, int]]:
        """Forward pass.
        
        Args:
            x: input images [B, 3, H, W]
            
        Returns:
            tokens: [B, N, embed_dim] where N = (H/patch_size) * (W/patch_size)
            grid_hw: (grid_h, grid_w) spatial dimensions
        """
        if self.backbone is not None:
            # Use timm backbone
            feats = self.backbone(x)[0]  # [B, C, H', W']
            b, c, h, w = feats.shape
            tokens = feats.flatten(2).transpose(1, 2)  # [B, H'*W', C]
            tokens = self.proj(tokens)  # [B, N, embed_dim]
            return tokens, (h, w)
        else:
            # Fallback conv
            x = self.proj(x)  # [B, embed_dim, H', W']
            b, c, h, w = x.shape
            tokens = x.flatten(2).transpose(1, 2)  # [B, N, embed_dim]
            return tokens, (h, w)
