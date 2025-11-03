# DeepSeek-OCR: Contexts Optical Compression

[![Paper](https://img.shields.io/badge/Paper-arXiv-red)](https://arxiv.org/abs/2510.18234)
[![License](https://img.shields.io/badge/License-MIT-blue.svg)](LICENSE)

A research implementation of **DeepSeek-OCR** based on the paper ["DeepSeek-OCR: Contexts Optical Compression"](https://arxiv.org/abs/2510.18234). This project explores OCR as a compression-decompression task using vision-language models.

## Overview

DeepSeek-OCR compresses text-rich images into compact vision tokens, then decodes them back to text using a language model. Key features:

- **High Compression**: 64-400 vision tokens for documents with 600-5000+ text tokens
- **Strong Performance**: ~97% accuracy with <10× compression ratio
- **Multi-Resolution**: Supports Tiny (512²), Small (640²), Base (1024²), Large (1280²) modes
- **Efficient**: Lower token count than traditional VLMs while maintaining quality

### Architecture

```
Image → DeepEncoder (SAM-like → Compressor 16× → CLIP-like) → LM Decoder → Text
        [Stage 1: Window Attn] [Conv 16× reduction] [Global Attn]   [GPT-2]
```

**DeepEncoder** (~380M params): Two-stage vision encoder with 16× compression
**Decoder** (~570M params): Language model for text generation

See [ARCHITECTURE.md](ARCHITECTURE.md) for detailed explanation.

## Installation

### Requirements

- Python 3.8+
- PyTorch 2.0+
- CUDA 11.8+ (for GPU)
- 16GB+ RAM (32GB+ recommended)

### Quick Install

```bash
# Clone repository
git clone https://github.com/yourusername/Deepseek-OCR.git
cd Deepseek-OCR

# Create virtual environment
conda create -n deepseek-ocr python=3.10 -y
conda activate deepseek-ocr

# Install dependencies
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements.txt
```

### Optional Dependencies

```bash
# For Tesseract baseline
sudo apt-get install tesseract-ocr  # Ubuntu
brew install tesseract  # macOS
pip install pytesseract

# For PaddleOCR baseline
pip install paddleocr

# For demo UI
pip install gradio
```

## Quick Start

### Inference

```bash
# Run OCR on single image
python scripts/infer.py --image path/to/image.jpg --mode base

# Process directory of images
python scripts/infer.py --image_dir path/to/images/ --output results.txt

# Use different resolution mode
python scripts/infer.py --image document.png --mode large
```

**Resolution Modes:**
- `tiny` (64 tokens): Fast, for simple documents
- `small` (100 tokens): Balanced quality/speed
- `base` (256 tokens): Standard documents (default)
- `large` (400 tokens): Complex layouts, best quality

### Python API

```python
import torch
from transformers import AutoTokenizer
from models.ocr_model import DeepSeekOCR
from utils.image_io import load_image

# Load model
model = DeepSeekOCR()
model = model.eval()
tokenizer = AutoTokenizer.from_pretrained("gpt2")

# Load and process image
image = load_image("document.jpg", mode="base")

# Generate text
with torch.no_grad():
    generated_ids = model.generate(
        images=image,
        max_length=2048,
        temperature=0.7,
        top_p=0.9
    )

text = tokenizer.decode(generated_ids[0], skip_special_tokens=True)
print(text)
```

### Web Demo

```bash
# Launch Gradio interface
python demo/app.py --checkpoint path/to/model.pt --port 7860

# Access at http://localhost:7860
```

## Evaluation

### Prepare Dataset

Organize your dataset:

```
dataset/
├── images/
│   ├── img001.jpg
│   ├── img002.png
│   └── ...
└── annotations.json  # {"img001.jpg": "ground truth text", ...}
```

### Run Evaluation

```bash
# Evaluate DeepSeek-OCR
python evaluation/evaluate_ocr.py \
    --dataset path/to/dataset \
    --checkpoint model.pt \
    --mode base \
    --output deepseek_results.json

# Tesseract baseline
python evaluation/baselines/tesseract_eval.py \
    --dataset path/to/dataset \
    --output tesseract_results.json

# PaddleOCR baseline
python evaluation/baselines/paddle_eval.py \
    --dataset path/to/dataset \
    --output paddleocr_results.json
```

### Metrics

- **CER (Character Error Rate)**: Character-level edit distance
- **WER (Word Error Rate)**: Word-level edit distance
- **Normalized Edit Distance**: Levenshtein distance normalized by length

### Expected Results (Paper)

On OmniDocBench:

| Model | Tokens/Page | CER | Notes |
|-------|------------|-----|-------|
| GOT-OCR2.0 | 256 | - | SOTA baseline |
| **DeepSeek-OCR (Small)** | **100** | - | Beats GOT-OCR2.0 |
| **DeepSeek-OCR (Large)** | **400** | - | Matches SOTA |
| MinerU2.0 | ~7000 | - | Token-heavy |
| **DeepSeek-OCR (Gundam)** | **<800** | - | Beats MinerU |

Compression Study (Fox Benchmark):

| Compression Ratio | Accuracy | Token Budget |
|------------------|----------|-------------|
| <10× | ~97% | 100-256 |
| 10-15× | 80-95% | 64-100 |
| ~20× | ~60% | 64 |

## 🔧 Training

### Stage 1: Train DeepEncoder

```bash
python scripts/train_encoder.py \
    --data_path path/to/ocr_data \
    --output_dir checkpoints/encoder \
    --epochs 2 \
    --batch_size 32 \
    --lr 5e-5
```

### Stage 2: Train Full Model

```bash
python scripts/train_ocr.py \
    --encoder_checkpoint checkpoints/encoder/best.pt \
    --data_path path/to/ocr_data \
    --output_dir checkpoints/full_model \
    --epochs 5 \
    --batch_size 16 \
    --lr 3e-5
```

**Training Data Mix (Paper):**
- 70% OCR data (layout + non-layout)
- 20% General vision data
- 10% Text-only data

**Note**: This implementation provides the architecture. Training requires large-scale data and compute similar to the paper (20 nodes × 8 A100-40G).

## Project Structure

```
Deepseek-OCR/
├── configs/              # Model and mode configurations
│   ├── model.yaml
│   └── modes.yaml
├── models/               # Model implementations
│   ├── deepencoder/
│   │   ├── sam_stage.py      # Window attention stage
│   │   ├── compressor.py     # 16× token compressor
│   │   ├── clip_stage.py     # Global attention stage
│   │   └── deepencoder.py    # Full encoder
│   ├── decoders/
│   │   ├── lm_decoder.py     # LM decoder
│   │   └── moe_layers.py     # MoE (optional)
│   └── ocr_model.py          # End-to-end model
├── utils/                # Utilities
│   ├── image_io.py          # Image loading & preprocessing
│   ├── metrics.py           # CER, WER metrics
│   ├── prompts.py           # OCR prompts
│   └── ...
├── scripts/              # Training & inference
│   ├── infer.py
│   ├── train_encoder.py
│   └── train_ocr.py
├── evaluation/           # Evaluation scripts
│   ├── evaluate_ocr.py
│   └── baselines/
│       ├── tesseract_eval.py
│       └── paddle_eval.py
├── demo/                 # Demo application
│   └── app.py
├── data/                 # Data directory
└── docs/                 # Documentation
    └── ARCHITECTURE.md
```

## Paper Summary

### Key Contributions

1. **Contexts Optical Compression**: First systematic study of compressing long text contexts via vision pathway
2. **DeepEncoder Design**: Two-stage encoder (window + global attention) with 16× compression
3. **Compression-Accuracy Analysis**: Quantifies feasibility of vision-based context compression
4. **Practical Performance**: Matches/exceeds SOTA OCR with far fewer tokens

### Innovations

- **Low-Activation Design**: Window attention + compression before global attention
- **Multi-Resolution Support**: Single model handles 512² to 1280² (+ tiling for ultra-high-res)
- **Token Efficiency**: 100-400 tokens vs 1000s in traditional approaches
- **Research Applications**: Context compression, memory forgetting mechanisms in LLMs

### Citation

```bibtex
@article{wei2025deepseek,
  title={DeepSeek-OCR: Contexts Optical Compression},
  author={Wei, Haoran and Sun, Yaofeng and Li, Yukun},
  journal={arXiv preprint arXiv:2510.18234},
  year={2025}
}
```

## Implementation Notes

### Differences from Paper

This is a **research-friendly implementation** that maintains architectural fidelity while being compute-accessible:

| Component | Paper | Our Implementation |
|-----------|-------|-------------------|
| SAM Stage | SAM-base (80M) | Swin-Tiny (timm) |
| CLIP Stage | CLIP-large (300M) | CLIP-base (HF) |
| Decoder | DeepSeek-3B-MoE | GPT-2 (HF) |
| Training Scale | 20×8 A100-40G | Single GPU friendly |
| Total Params | ~950M (~570M active) | ~300M |

### Design Choices

1. **Compute-Friendly Backbones**: Use smaller but architecturally similar models
2. **HuggingFace Integration**: Leverage pretrained CLIP and LM weights
3. **Modular Design**: Easy to swap components (e.g., upgrade to larger decoder)
4. **Fallbacks**: Simple implementations when optional deps (timm, HF) unavailable

## Development

### Testing

```bash
# Test model forward pass
python -c "from models.ocr_model import DeepSeekOCR; model = DeepSeekOCR(); print('Model loaded:', model.get_num_params())"

# Test image loading
python -c "from utils.image_io import load_image; img = load_image('test.jpg', 'base'); print('Shape:', img.shape)"
```

### Adding Custom Backbones

Replace components in `models/deepencoder/`:

```python
# Example: Use different SAM-like backbone
from models.deepencoder.sam_stage import SAMLikeStage

stage1 = SAMLikeStage(
    model_name="swin_base_patch4_window7_224",  # Larger Swin
    embed_dim=256,
    pretrained=True
)
```

## Resources

- **Paper**: [arXiv:2510.18234](https://arxiv.org/abs/2510.18234)
- **Official Repo**: [deepseek-ai/DeepSeek-OCR](https://github.com/deepseek-ai/DeepSeek-OCR)
- **Model Weights**: [HuggingFace](https://huggingface.co/deepseek-ai/DeepSeek-OCR)
- **Related Work**:
  - [DeepSeek-VL](https://arxiv.org/abs/2403.05525) - Vision-Language foundation
  - [Vary](https://arxiv.org/abs/2312.06109) - Vision vocabulary
  - [InternVL](https://arxiv.org/abs/2404.16821) - Multi-resolution VLM

## Contributing

Contributions welcome! Areas:

- [ ] Training scripts with distributed support
- [ ] Gundam (tiling) mode implementation
- [ ] Layout-aware OCR prompts
- [ ] Multilingual evaluation
- [ ] MoE decoder integration
- [ ] More baseline comparisons

## License

MIT License. See [LICENSE](LICENSE) for details.

## Acknowledgments

- DeepSeek AI team for the original paper and research
- HuggingFace for transformers library
- timm library for vision backbones
- Open-source OCR community (Tesseract, PaddleOCR)

---

**Note**: This is a research implementation for educational purposes. For production use, consider the [official DeepSeek-OCR model](https://huggingface.co/deepseek-ai/DeepSeek-OCR) or training on large-scale data


Star this repo if you find it useful :)

Follow me on [X (Twitter)](https://www.x.com/techwith_ram)