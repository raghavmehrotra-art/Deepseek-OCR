# DeepSeek-OCR Architecture

This document explains the architecture and implementation of DeepSeek-OCR based on the paper [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234).

## Overview

DeepSeek-OCR is an end-to-end Vision-Language Model (VLM) that performs OCR through **contexts optical compression**:
- **Input**: High-resolution images (512×512 to 1280×1280)
- **Compressed Representation**: 64-400 vision tokens
- **Output**: Text sequences (up to 2048 tokens)

The model achieves ~97% accuracy with <10× compression ratio and ~60% accuracy at 20× compression.

## Architecture Components

### 1. DeepEncoder (~380M parameters)

The encoder compresses images into compact vision tokens through three stages:

#### Stage 1: SAM-like (Window Attention)
- **Purpose**: Extract local features efficiently at high resolution
- **Implementation**: Swin Transformer backbone (window attention)
- **Parameters**: ~80M
- **Input**: [B, 3, H, W] RGB image
- **Output**: [B, N, 256] tokens where N = (H/16) × (W/16)
- **Why**: Window attention reduces activation memory compared to global attention

```python
# Example: 1024×1024 image
# Output: [B, 4096, 256] tokens (64×64 grid)
```

#### Compression Module (16× reduction)
- **Purpose**: Drastically reduce token count before expensive global attention
- **Implementation**: Two strided conv layers (k=3, s=2, p=1)
- **Compression**: 16× spatial reduction (4× per layer)
- **Channels**: 256 → 512 → 1024
- **Why**: Makes global attention tractable by reducing from 4096 to 256 tokens

```python
# Example: 4096 tokens → 256 tokens
# Conv1: [B, 256, 64, 64] → [B, 512, 32, 32]
# Conv2: [B, 512, 32, 32] → [B, 1024, 16, 16]
# Flatten: [B, 256, 1024]
```

#### Stage 2: CLIP-like (Global Attention)
- **Purpose**: Inject vision-language knowledge and global context
- **Implementation**: CLIP ViT encoder (without patch embedding)
- **Parameters**: ~300M
- **Input**: [B, 256, 1024] compressed tokens
- **Output**: [B, 256, 1024] globally-attended tokens
- **Why**: CLIP provides semantic understanding from pretraining

### 2. Decoder (~570M active parameters)

The decoder reconstructs text from compressed vision tokens:

#### LM Decoder
- **Implementation**: GPT-2 / OPT / LLaMA (HuggingFace)
- **Paper**: DeepSeek-3B-MoE (6/64 experts + 2 shared)
- **Our prototype**: GPT-2 (single expert for simplicity)
- **Function**: Maps vision tokens to text via causal language modeling

```python
# Concatenate vision and text tokens
inputs = [vision_tokens, text_embeddings]  # [B, N+L, d]
# Autoregressive generation
output = decoder(inputs)  # [B, N+L, vocab_size]
```

## Multi-Resolution Support

### Native Modes

| Mode  | Resolution   | Tokens | Padding | Use Case                |
|-------|-------------|--------|---------|-------------------------|
| Tiny  | 512×512     | 64     | No      | Simple documents        |
| Small | 640×640     | 100    | No      | Balanced quality/speed  |
| Base  | 1024×1024   | 256    | Yes     | Standard documents      |
| Large | 1280×1280   | 400    | Yes     | Complex layouts         |

### Aspect Ratio Preservation

Base and Large modes preserve aspect ratio via padding:
1. Scale image to fit within target size
2. Pad with white to exact size
3. Calculate valid token count: `tokens × (scaled_w/target_w) × (scaled_h/target_h)`

## Data Flow

```
Input Image [B, 3, 1024, 1024]
    ↓
SAM Stage (Swin-T) → [B, 4096, 256]
    ↓
Compressor (Conv 16×) → [B, 256, 1024]
    ↓
CLIP Stage (ViT) → [B, 256, 1024]
    ↓
Vision Projection → [B, 256, 768]
    ↓
LM Decoder (GPT-2) → [B, L, vocab_size]
    ↓
Generated Text: "Hello world..."
```

## Key Innovations

### 1. Contexts Optical Compression
- Treats OCR as compress-decompress task
- Vision tokens = compressed representation of text
- Enables efficient long-context processing

### 2. Two-Stage Encoder Design
- **Stage 1 (SAM)**: Local features, low activation
- **Compressor**: 16× token reduction
- **Stage 2 (CLIP)**: Global features, semantic knowledge
- **Result**: High compression with preserved semantics

### 3. Compression-Accuracy Tradeoff
| Compression Ratio | Accuracy | Use Case |
|------------------|----------|----------|
| <10×             | ~97%     | Production OCR |
| 10-20×           | 60-97%   | Context compression research |
| >20×             | <60%     | Forgetting mechanism studies |

## Training Pipeline (Paper)

### Stage A: DeepEncoder Pretraining
- **Objective**: Next token prediction with small LM
- **Data**: OCR data + 100M LAION captions
- **Duration**: 2 epochs
- **Batch size**: 1280
- **Optimizer**: AdamW, lr=5e-5, cosine schedule

### Stage B: End-to-End Training
- **Data mix**: 70% OCR, 20% vision, 10% text
- **Sequence length**: 8192 tokens
- **Pipeline parallelism**: 4 stages (SAM+Comp frozen in PP0)
- **Scale**: 20 nodes × 8 A100-40G

## Our Implementation

We provide a **compute-friendly** implementation:

| Component | Paper (Faithful) | Our Implementation |
|-----------|-----------------|-------------------|
| SAM Stage | SAM-base 80M | Swin-Tiny (timm) |
| CLIP Stage | CLIP-large 300M | CLIP-base (HF) |
| Decoder | DeepSeek-3B-MoE | GPT-2 (HF) |
| Total params | ~950M (~570M active) | ~300M |
| Training data | 100M+ samples | User-provided |

This allows local prototyping while maintaining the paper's architectural concepts.

## Performance Expectations

### Paper Results (OmniDocBench)
- 100 tokens (small mode): Beats GOT-OCR2.0 (256 tokens)
- 400 tokens (large mode): Matches state-of-the-art
- <800 tokens (Gundam): Beats MinerU2.0 (~7000 tokens)

### Our Implementation
- Architecture faithful to paper
- Performance depends on training data and compute
- Suitable for research and prototyping

## Extensions

1. **Gundam Mode (Tiling)**: For ultra-high-res documents
   - n × 640×640 local views + 1 × 1024×1024 global view
   - Token count: n×100 + 256

2. **Layout-Aware OCR**: Preserve document structure
   - Special prompts for layout preservation

3. **Multilingual**: Supports ~100 languages (with proper training data)

4. **Deep Parsing**: Secondary model calls for charts/formulas

## References

- Paper: [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
- GitHub: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- Model: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
