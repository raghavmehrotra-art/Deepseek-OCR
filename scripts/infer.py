#!/usr/bin/env python3
"""Inference script for DeepSeek-OCR.

Performs OCR on images using the trained DeepSeek-OCR model.

Usage:
    python scripts/infer.py --image path/to/image.jpg --mode base
    python scripts/infer.py --image_dir path/to/images/ --output results.txt
"""
import argparse
import os
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

# Add parent directory to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ocr_model import DeepSeekOCR
from utils.image_io import load_image, MODE_CONFIGS
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Inference")
    
    # Input
    parser.add_argument("--image", type=str, help="Path to single image")
    parser.add_argument("--image_dir", type=str, help="Path to image directory")
    
    # Model
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--mode",
        type=str,
        default="base",
        choices=list(MODE_CONFIGS.keys()),
        help="Resolution mode"
    )
    
    # Generation
    parser.add_argument("--max_length", type=int, default=2048, help="Max text length")
    parser.add_argument("--temperature", type=float, default=0.7, help="Sampling temperature")
    parser.add_argument("--top_p", type=float, default=0.9, help="Nucleus sampling")
    
    # Output
    parser.add_argument("--output", type=str, default=None, help="Output file")
    parser.add_argument("--device", type=str, default="cuda" if torch.cuda.is_available() else "cpu")
    
    return parser.parse_args()


def load_model(checkpoint_path=None, device="cuda"):
    """Load DeepSeek-OCR model."""
    logger.info("Loading model...")
    
    # Create model with default config
    model = DeepSeekOCR()
    
    # Load checkpoint if provided
    if checkpoint_path and os.path.exists(checkpoint_path):
        logger.info(f"Loading checkpoint from {checkpoint_path}")
        state_dict = torch.load(checkpoint_path, map_location=device)
        model.load_state_dict(state_dict)
    else:
        logger.warning("No checkpoint provided, using randomly initialized model")
    
    model = model.to(device)
    model.eval()
    
    # Load tokenizer
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    logger.info(f"Model loaded with {model.get_num_params()['total']:,} parameters")
    
    return model, tokenizer


def run_ocr(image_path, model, tokenizer, mode="base", device="cuda", **gen_kwargs):
    """Run OCR on a single image."""
    # Load and preprocess image
    image = load_image(image_path, mode=mode)
    image = image.to(device)
    
    # Generate text
    with torch.no_grad():
        generated_ids = model.generate(
            images=image,
            **gen_kwargs
        )
    
    # Decode
    text = tokenizer.batch_decode(generated_ids, skip_special_tokens=True)[0]
    
    return text


def main():
    args = parse_args()
    
    # Validate input
    if not args.image and not args.image_dir:
        logger.error("Must provide either --image or --image_dir")
        return
    
    # Load model
    model, tokenizer = load_model(args.checkpoint, args.device)
    
    # Generation kwargs
    gen_kwargs = {
        "max_length": args.max_length,
        "temperature": args.temperature,
        "top_p": args.top_p
    }
    
    results = []
    
    # Process single image
    if args.image:
        logger.info(f"Processing image: {args.image}")
        text = run_ocr(
            args.image,
            model,
            tokenizer,
            mode=args.mode,
            device=args.device,
            **gen_kwargs
        )
        print(f"\n{'='*60}")
        print(f"Image: {args.image}")
        print(f"Mode: {args.mode} ({MODE_CONFIGS[args.mode]['tokens']} tokens)")
        print(f"{'='*60}")
        print(text)
        print(f"{'='*60}\n")
        results.append((args.image, text))
    
    # Process directory
    elif args.image_dir:
        image_dir = Path(args.image_dir)
        image_files = list(image_dir.glob("*.jpg")) + list(image_dir.glob("*.png"))
        
        logger.info(f"Processing {len(image_files)} images from {args.image_dir}")
        
        for img_path in image_files:
            logger.info(f"Processing {img_path.name}...")
            try:
                text = run_ocr(
                    str(img_path),
                    model,
                    tokenizer,
                    mode=args.mode,
                    device=args.device,
                    **gen_kwargs
                )
                results.append((str(img_path), text))
                print(f"{img_path.name}: {len(text)} chars")
            except Exception as e:
                logger.error(f"Error processing {img_path.name}: {e}")
    
    # Save results
    if args.output and results:
        logger.info(f"Saving results to {args.output}")
        with open(args.output, "w", encoding="utf-8") as f:
            for img_path, text in results:
                f.write(f"Image: {img_path}\n")
                f.write(f"{text}\n")
                f.write("\n" + "="*60 + "\n\n")
        logger.info("Done!")


if __name__ == "__main__":
    main()
