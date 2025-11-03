"""Token Compressor: 16x token reduction via convolutional downsampling.

Reduces the number of vision tokens by 16x before global attention to
control activation memory while maintaining semantic information.

Paper reference: Section 3.2.1 - Between SAM and CLIP, a 2-layer conv module
performs 16x downsampling. Each conv has kernel=3, stride=2, padding=1.
Channels increase from 256 to 1024.
"""
import torch
import torch.nn as nn


class ConvCompressor(nn.Module):
    """Convolutional token compressor for DeepEncoder.
    
    Applies two strided convolutions to reduce token count by 16x:
    - Conv1: stride=2, reduces by 4x (2x2)
    - Conv2: stride=2, reduces by 4x (2x2)
    - Total: 16x compression
    
    Args:
        in_ch: input channel dimension (from SAM stage)
        out_ch: output channel dimension (for CLIP stage)
        hidden_ch: hidden channel dimension (default: in_ch * 2)
    """
    def __init__(self, in_ch: int = 256, out_ch: int = 1024, hidden_ch: int = None):
        super().__init__()
        if hidden_ch is None:
            hidden_ch = in_ch * 2
        
        # First compression layer: 256 -> 512, spatial /4
        self.conv1 = nn.Conv2d(
            in_ch, hidden_ch,
            kernel_size=3, stride=2, padding=1
        )
        self.norm1 = nn.LayerNorm(hidden_ch)
        self.act1 = nn.GELU()
        
        # Second compression layer: 512 -> 1024, spatial /4
        self.conv2 = nn.Conv2d(
            hidden_ch, out_ch,
            kernel_size=3, stride=2, padding=1
        )
        self.norm2 = nn.LayerNorm(out_ch)
        self.act2 = nn.GELU()
    
    def forward(
        self,
        tokens: torch.Tensor,
        grid_hw: tuple[int, int]
    ) -> tuple[torch.Tensor, tuple[int, int]]:
        """Compress tokens by 16x.
        
        Args:
            tokens: [B, N, C] input tokens from SAM stage
            grid_hw: (H, W) spatial grid dimensions
            
        Returns:
            compressed_tokens: [B, N/16, out_ch]
            new_grid_hw: (H/4, W/4) new spatial dimensions
        """
        h, w = grid_hw
        b, n, c = tokens.shape
        
        # Reshape to spatial layout
        x = tokens.transpose(1, 2).reshape(b, c, h, w)
        
        # First compression
        x = self.conv1(x)  # [B, hidden_ch, H/2, W/2]
        x = x.permute(0, 2, 3, 1)  # [B, H/2, W/2, hidden_ch]
        x = self.norm1(x)
        x = self.act1(x)
        x = x.permute(0, 3, 1, 2)  # [B, hidden_ch, H/2, W/2]
        
        # Second compression
        x = self.conv2(x)  # [B, out_ch, H/4, W/4]
        x = x.permute(0, 2, 3, 1)  # [B, H/4, W/4, out_ch]
        x = self.norm2(x)
        x = self.act2(x)
        x = x.permute(0, 3, 1, 2)  # [B, out_ch, H/4, W/4]
        
        # Flatten to tokens
        b, c, new_h, new_w = x.shape
        compressed = x.flatten(2).transpose(1, 2)  # [B, N/16, out_ch]
        
        return compressed, (new_h, new_w)
