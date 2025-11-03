"""DeepSeek-OCR: End-to-end OCR model with contexts optical compression.

Combines DeepEncoder (vision) and LM decoder to perform OCR through
compression-decompression of visual contexts.

Paper reference: Section 3.1 - DeepSeek-OCR is an end-to-end VLM with
DeepEncoder (~380M) + DeepSeek-3B-MoE decoder (~570M active).
"""
import torch
import torch.nn as nn
from typing import Optional, Dict, Any
from .deepencoder.deepencoder import DeepEncoder
from .decoders.lm_decoder import LMDecoder


class DeepSeekOCR(nn.Module):
    """DeepSeek-OCR: Unified encoder-decoder for optical character recognition.
    
    Architecture:
        Image → DeepEncoder → Compressed Vision Tokens → LM Decoder → Text
    
    Args:
        encoder_config: configuration dict for DeepEncoder
        decoder_config: configuration dict for LMDecoder
    """
    def __init__(
        self,
        encoder_config: Optional[Dict[str, Any]] = None,
        decoder_config: Optional[Dict[str, Any]] = None
    ):
        super().__init__()
        
        # Default configs
        if encoder_config is None:
            encoder_config = {
                "embed_dim": 256,
                "clip_dim": 1024,
                "sam_model": "swin_tiny_patch4_window7_224",
                "clip_model": "openai/clip-vit-base-patch32",
                "use_pretrained": True
            }
        
        if decoder_config is None:
            decoder_config = {
                "d_latent": 1024,
                "model_name": "gpt2",
                "use_pretrained": True,
                "freeze_lm": False
            }
        
        # Initialize encoder and decoder
        self.encoder = DeepEncoder(**encoder_config)
        self.decoder = LMDecoder(**decoder_config)
    
    def forward(
        self,
        images: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None,
        return_vision_tokens: bool = False
    ) -> torch.Tensor:
        """Forward pass for training.
        
        Args:
            images: [B, 3, H, W] input images
            input_ids: [B, L] text token ids for teacher forcing
            attention_mask: [B, L] attention mask
            labels: [B, L] target labels
            return_vision_tokens: return vision tokens in output
            
        Returns:
            loss or logits, optionally with vision tokens
        """
        # Encode images to compressed vision tokens
        vision_tokens = self.encoder(images)
        
        # Decode to text
        output = self.decoder(
            vision_tokens=vision_tokens,
            input_ids=input_ids,
            attention_mask=attention_mask,
            labels=labels
        )
        
        if return_vision_tokens:
            return output, vision_tokens
        return output
    
    def generate(
        self,
        images: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """Generate text from images.
        
        Args:
            images: [B, 3, H, W] input images
            max_length: maximum generation length
            temperature: sampling temperature
            top_p: nucleus sampling parameter
            
        Returns:
            generated_ids: [B, L] generated token ids
        """
        # Encode images
        vision_tokens = self.encoder(images)
        
        # Generate text
        generated_ids = self.decoder.generate(
            vision_tokens=vision_tokens,
            max_length=max_length,
            temperature=temperature,
            top_p=top_p,
            **kwargs
        )
        
        return generated_ids
    
    @torch.no_grad()
    def ocr(
        self,
        images: torch.Tensor,
        tokenizer,
        max_length: int = 2048,
        mode: str = "free"
    ) -> list[str]:
        """Perform OCR on images.
        
        Args:
            images: [B, 3, H, W] input images
            tokenizer: HuggingFace tokenizer
            max_length: maximum text length
            mode: OCR mode ('free' or 'layout')
            
        Returns:
            list of OCR text strings
        """
        # Generate token ids
        generated_ids = self.generate(
            images=images,
            max_length=max_length,
            temperature=0.7,
            top_p=0.9
        )
        
        # Decode to text
        texts = tokenizer.batch_decode(
            generated_ids,
            skip_special_tokens=True
        )
        
        return texts
    
    def get_num_params(self) -> Dict[str, int]:
        """Get parameter counts."""
        encoder_params = sum(p.numel() for p in self.encoder.parameters())
        decoder_params = sum(p.numel() for p in self.decoder.parameters())
        total_params = encoder_params + decoder_params
        
        return {
            "encoder": encoder_params,
            "decoder": decoder_params,
            "total": total_params
        }
