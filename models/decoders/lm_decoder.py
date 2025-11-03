"""Language Model Decoder for DeepSeek-OCR.

Maps compressed vision tokens to text sequences using a transformer-based
language model (optionally MoE-enhanced).

Paper reference: Section 3.3 - Uses DeepSeek-3B-MoE with 570M activated
parameters (6/64 routed experts + 2 shared experts).
"""
import torch
import torch.nn as nn
from typing import Optional

try:
    from transformers import AutoModelForCausalLM, AutoConfig
    HF_AVAILABLE = True
except ImportError:
    HF_AVAILABLE = False


class LMDecoder(nn.Module):
    """Language Model decoder for text generation from vision tokens.
    
    Uses a pretrained or randomly initialized language model to decode
    compressed vision tokens into text sequences.
    
    Args:
        d_latent: dimension of vision tokens from encoder
        model_name: HF model name (e.g., 'gpt2', 'facebook/opt-350m')
        use_pretrained: load pretrained LM weights
        freeze_lm: freeze language model weights during training
    """
    def __init__(
        self,
        d_latent: int = 1024,
        model_name: str = "gpt2",
        use_pretrained: bool = True,
        freeze_lm: bool = False
    ):
        super().__init__()
        self.d_latent = d_latent
        
        if HF_AVAILABLE:
            # Load language model
            if use_pretrained:
                self.lm = AutoModelForCausalLM.from_pretrained(model_name)
            else:
                config = AutoConfig.from_pretrained(model_name)
                self.lm = AutoModelForCausalLM.from_config(config)
            
            self.hidden_size = self.lm.config.hidden_size
            self.vocab_size = self.lm.config.vocab_size
            
            # Vision token projection to LM hidden size
            self.vision_proj = nn.Linear(d_latent, self.hidden_size)
            
            # Freeze LM if requested
            if freeze_lm:
                for param in self.lm.parameters():
                    param.requires_grad = False
        else:
            print("Warning: transformers not available, using simple decoder")
            self.hidden_size = d_latent
            self.vocab_size = 32000
            self.vision_proj = nn.Identity()
            self.lm = nn.TransformerDecoder(
                nn.TransformerDecoderLayer(
                    d_model=d_latent,
                    nhead=8,
                    dim_feedforward=d_latent * 4,
                    batch_first=True
                ),
                num_layers=6
            )
            self.lm_head = nn.Linear(d_latent, self.vocab_size)
    
    def forward(
        self,
        vision_tokens: torch.Tensor,
        input_ids: Optional[torch.Tensor] = None,
        attention_mask: Optional[torch.Tensor] = None,
        labels: Optional[torch.Tensor] = None
    ) -> torch.Tensor:
        """Forward pass through decoder.
        
        Args:
            vision_tokens: [B, N, d_latent] compressed vision features
            input_ids: [B, L] text token ids (for teacher forcing)
            attention_mask: [B, L] attention mask
            labels: [B, L] target labels for loss computation
            
        Returns:
            logits: [B, L, vocab_size] or loss if labels provided
        """
        # Project vision tokens to LM hidden size
        vision_embeds = self.vision_proj(vision_tokens)  # [B, N, hidden_size]
        
        if HF_AVAILABLE:
            # Get text embeddings if input_ids provided
            if input_ids is not None:
                text_embeds = self.lm.get_input_embeddings()(input_ids)
                # Concatenate vision and text embeddings
                inputs_embeds = torch.cat([vision_embeds, text_embeds], dim=1)
                
                # Extend attention mask for vision tokens
                if attention_mask is not None:
                    vision_mask = torch.ones(
                        vision_embeds.shape[:2],
                        dtype=attention_mask.dtype,
                        device=attention_mask.device
                    )
                    attention_mask = torch.cat([vision_mask, attention_mask], dim=1)
            else:
                inputs_embeds = vision_embeds
                attention_mask = None
            
            # Forward through LM
            outputs = self.lm(
                inputs_embeds=inputs_embeds,
                attention_mask=attention_mask,
                labels=labels,
                return_dict=True
            )
            
            if labels is not None:
                return outputs.loss
            return outputs.logits
        else:
            # Fallback simple decoder
            memory = vision_embeds
            if input_ids is not None:
                tgt = self.lm.get_input_embeddings()(input_ids)
            else:
                tgt = vision_embeds
            
            output = self.lm(tgt, memory)
            logits = self.lm_head(output)
            return logits
    
    def generate(
        self,
        vision_tokens: torch.Tensor,
        max_length: int = 512,
        temperature: float = 1.0,
        top_p: float = 0.9,
        **kwargs
    ) -> torch.Tensor:
        """Generate text from vision tokens.
        
        Args:
            vision_tokens: [B, N, d_latent] compressed vision features
            max_length: maximum generation length
            temperature: sampling temperature
            top_p: nucleus sampling parameter
            
        Returns:
            generated_ids: [B, L] generated token ids
        """
        vision_embeds = self.vision_proj(vision_tokens)
        
        if HF_AVAILABLE and hasattr(self.lm, 'generate'):
            # Use HF generation
            outputs = self.lm.generate(
                inputs_embeds=vision_embeds,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p,
                do_sample=True,
                **kwargs
            )
            return outputs
        else:
            # Simple greedy decoding
            batch_size = vision_tokens.shape[0]
            device = vision_tokens.device
            
            # Start with BOS token (assume 1)
            generated = torch.ones(batch_size, 1, dtype=torch.long, device=device)
            
            for _ in range(max_length - 1):
                logits = self.forward(vision_tokens, generated)
                next_token = logits[:, -1, :].argmax(dim=-1, keepdim=True)
                generated = torch.cat([generated, next_token], dim=1)
                
                # Stop if all sequences generated EOS (assume 2)
                if (next_token == 2).all():
                    break
            
            return generated
