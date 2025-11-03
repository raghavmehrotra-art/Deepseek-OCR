#!/usr/bin/env python3
"""PaddleOCR baseline evaluation.

Evaluates PaddleOCR as a baseline for comparison.

Usage:
    python evaluation/baselines/paddle_eval.py --dataset path/to/dataset
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    from paddleocr import PaddleOCR
    PADDLE_AVAILABLE = True
except ImportError:
    PADDLE_AVAILABLE = False
    print("Warning: paddleocr not installed. Install with: pip install paddleocr")

from utils.metrics import cer, wer
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="PaddleOCR Baseline Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="en",
        help="Language code"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="paddleocr_results.json",
        help="Output results file"
    )
    return parser.parse_args()


def load_dataset(dataset_path: str) -> List[Tuple[str, str]]:
    """Load dataset."""
    dataset_path = Path(dataset_path)
    images_dir = dataset_path / "images"
    annotations_file = dataset_path / "annotations.json"
    
    with open(annotations_file, "r", encoding="utf-8") as f:
        annotations = json.load(f)
    
    dataset = []
    for img_name, gt_text in annotations.items():
        img_path = images_dir / img_name
        if img_path.exists():
            dataset.append((str(img_path), gt_text))
    
    return dataset


def run_paddleocr(
    dataset: List[Tuple[str, str]],
    lang: str = "en"
) -> Dict:
    """Run PaddleOCR on dataset."""
    if not PADDLE_AVAILABLE:
        raise RuntimeError("paddleocr not available")
    
    # Initialize PaddleOCR
    ocr = PaddleOCR(use_angle_cls=True, lang=lang, show_log=False)
    
    cer_scores = []
    wer_scores = []
    predictions = []
    
    for img_path, gt_text in tqdm(dataset, desc="Running PaddleOCR"):
        # Run PaddleOCR
        result = ocr.ocr(img_path, cls=True)
        
        # Extract text
        if result and result[0]:
            pred_text = " ".join([line[1][0] for line in result[0]])
        else:
            pred_text = ""
        
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
    
    results = {
        "method": "paddleocr",
        "lang": lang,
        "num_samples": len(dataset),
        "avg_cer": sum(cer_scores) / len(cer_scores),
        "avg_wer": sum(wer_scores) / len(wer_scores),
        "predictions": predictions
    }
    
    return results


def main():
    args = parse_args()
    
    # Load dataset
    dataset = load_dataset(args.dataset)
    logger.info(f"Loaded {len(dataset)} samples")
    
    # Run PaddleOCR
    results = run_paddleocr(dataset, args.lang)
    
    # Print results
    print("\n" + "="*60)
    print("PADDLEOCR BASELINE RESULTS")
    print("="*60)
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
