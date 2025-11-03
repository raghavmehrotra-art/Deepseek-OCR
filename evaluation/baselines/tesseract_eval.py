#!/usr/bin/env python3
"""Tesseract OCR baseline evaluation.

Evaluates Tesseract OCR as a baseline for comparison.

Usage:
    python evaluation/baselines/tesseract_eval.py --dataset path/to/dataset
"""
import argparse
import json
import sys
from pathlib import Path
from typing import List, Tuple, Dict
from tqdm import tqdm

sys.path.insert(0, str(Path(__file__).parent.parent.parent))

try:
    import pytesseract
    from PIL import Image
    TESSERACT_AVAILABLE = True
except ImportError:
    TESSERACT_AVAILABLE = False
    print("Warning: pytesseract not installed. Install with: pip install pytesseract")

from utils.metrics import cer, wer
from utils.logging import get_logger

logger = get_logger(__name__)


def parse_args():
    parser = argparse.ArgumentParser(description="Tesseract Baseline Evaluation")
    parser.add_argument(
        "--dataset",
        type=str,
        required=True,
        help="Path to dataset directory"
    )
    parser.add_argument(
        "--lang",
        type=str,
        default="eng",
        help="Tesseract language"
    )
    parser.add_argument(
        "--output",
        type=str,
        default="tesseract_results.json",
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


def run_tesseract(
    dataset: List[Tuple[str, str]],
    lang: str = "eng"
) -> Dict:
    """Run Tesseract OCR on dataset."""
    if not TESSERACT_AVAILABLE:
        raise RuntimeError("pytesseract not available")
    
    cer_scores = []
    wer_scores = []
    predictions = []
    
    for img_path, gt_text in tqdm(dataset, desc="Running Tesseract"):
        # Run Tesseract
        img = Image.open(img_path)
        pred_text = pytesseract.image_to_string(img, lang=lang)
        
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
        "method": "tesseract",
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
    
    # Run Tesseract
    results = run_tesseract(dataset, args.lang)
    
    # Print results
    print("\n" + "="*60)
    print("TESSERACT BASELINE RESULTS")
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
