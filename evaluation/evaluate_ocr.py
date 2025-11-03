#!/usr/bin/env python3
"""Evaluation script for DeepSeek-OCR.

Evaluates OCR performance using CER, WER, and edit distance metrics.

Usage:
    python evaluation/evaluate_ocr.py --dataset path/to/dataset --checkpoint model.pt
"""
import argparse
import json
import os
import sys
from pathlib import Path
from typing import List, Tuple, Dict
import torch
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent))

from models.ocr_model import DeepSeekOCR
from utils.image_io import load_image
from utils.metrics import cer, wer
from utils.logging import get_logger
from transformers import AutoTokenizer

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Evaluate DeepSeek-OCR")
    
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory (images/ and annotations.json)"
    )
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
        help="Resolution mode"
    )
    parser.add_argument(
        "--device",
        type=str,
        default="cuda" if torch.cuda.is_available() else "cpu"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="evaluation_results.json",
        help="Output results file"
    )
    
    return parser.parse_args()


def load_dataset(dataset_path: str) -> List[Tuple[str, str]]:
    """Load dataset with image paths and ground truth text.
    
    Expected structure:
        dataset_path/
            images/
                img1.jpg
                img2.png
            annotations.json  # {"img1.jpg": "ground truth text", ...}
    """
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    annotations_file = dataset_path / "annotations.json"
    
    # Load annotations
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    # Build dataset
    dataset = []
    for img_name, gt_text in annotations.items():
        img_path = images_dir / img_name
        if img_path.exists():
            dataset.append((str(img_path), gt_text))
        else:
            logger.warning(f"Image not found: {img_path}")
    
    logger.info(f"Loaded {len(dataset)} samples")
    return dataset


def evaluate(
    model,
    tokenizer,
    dataset: List[Tuple[str, str]],
    mode: str = "base",
    device: str = "cuda"
) -> Dict:
    """Evaluate model on dataset."""
    model.eval()
    
    cer_scores = []
    wer_scores = []
    predictions = []
    
    with torch.no_grad():
        for img_path, gt_text in tqdm(dataset, desc="Evaluating"):
            # Load image
            image = load_image(img_path, mode=mode).to(device)
            
            # Generate prediction
            generated_ids = model.generate(
                images=image,
                max_length=2048,
                temperature=0.7,
                top_p=0.9
            )
            pred_text = tokenizer.batch_decode(
                generated_ids,
                skip_special_tokens=True
            )[0]
            
            # Calculate metrics
            cer_score = cer(gt_text, pred_text)
            wer_score = wer(gt_text, pred_text)
            
            cer_scores.append(cer_score)
            wer_scores.append(wer_score)
            predictions.append({
                "image": img_path,
                "ground_truth": gt_text,
                "prediction": pred_text,
                "cer": cer_score,
                "wer": wer_score
            })
    
    # Aggregate results
    results = {
        "num_samples": len(dataset),
        "mode": mode,
        "avg_cer": sum(cer_scores) / len(cer_scores) if cer_scores else 0,
        "avg_wer": sum(wer_scores) / len(wer_scores) if wer_scores else 0,
        "predictions": predictions
    }
    
    return results


def main():
    args = parse_args()
    
    # Load model
    logger.info("Loading model...")
    model = DeepSeekOCR()
    
    if args.checkpoint and os.path.exists(args.checkpoint):
        state_dict = torch.load(args.checkpoint, map_location=args.device)
        model.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from {args.checkpoint}")
    else:
        logger.warning("No checkpoint provided, using random weights")
    
    model = model.to(args.device)
    tokenizer = AutoTokenizer.from_pretrained("gpt2")
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    
    # Evaluate
    logger.info("Starting evaluation...")
    results = evaluate(model, tokenizer, dataset, args.mode, args.device)
    
    # Print summary
    print("\n" + "="*60)
    print("EVALUATION RESULTS")
    print("="*60)
    print(f"Dataset: {args.dataset}")
    print(f"Mode: {results['mode']}")
    print(f"Samples: {results['num_samples']}")
    print(f"Average CER: {results['avg_cer']:.4f}")
    print(f"Average WER: {results['avg_wer']:.4f}")
    print("="*60 + "\n")
    
    # Save results
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(results, f, indent=2, ensure_ascii=False)
    logger.info(f"Results saved to {args.output}")


if __name__ == "__main__":
    main()
