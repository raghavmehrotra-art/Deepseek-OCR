#!/usr/bin/env python3
"""Gradio demo app for DeepSeek-OCR.

Provides a web interface for running OCR on uploaded images.

Usage:
    python demo/app.py --checkpoint path/to/model.pt
"""
import argparse
import sys
from pathlib import Path
import torch
from transformers import AutoTokenizer

sys.path.insert(0, str(Path(__file__).parent.parent))

try:
    import gradio as gr
    GRADIO_AVAILABLE = True
except ImportError:
    GRADIO_AVAILABLE = False
    print("Gradio not installed. Install with: pip install gradio")

from models.ocr_model import DeepSeekOCR
from utils.image_io import load_image, MODE_CONFIGS
from utils.logging import get_logger

logger = get_logger(__name__)

# Global model and tokenizer
MODEL = None
TOKENIZER = None
DEVICE = "cuda" if torch.cuda.is_available() else "cpu"


def load_model(checkpoint_path=None):
    """Load model and tokenizer."""
    global MODEL, TOKENIZER
    
    logger.info("Loading model...")
    MODEL = DeepSeekOCR()
    
    if checkpoint_path:
        state_dict = torch.load(checkpoint_path, map_location=DEVICE)
        MODEL.load_state_dict(state_dict)
        logger.info(f"Loaded checkpoint from {checkpoint_path}")
    else:
        logger.warning("No checkpoint provided, using random weights")
    
    MODEL = MODEL.to(DEVICE)
    MODEL.eval()
    
    TOKENIZER = AutoTokenizer.from_pretrained("gpt2")
    if TOKENIZER.pad_token is None:
        TOKENIZER.pad_token = TOKENIZER.eos_token
    
    logger.info(f"Model loaded with {MODEL.get_num_params()['total']:,} parameters")


def run_ocr_demo(image, mode, temperature, top_p, max_length):
    """Run OCR on uploaded image."""
    if MODEL is None:
        return "Model not loaded!"
    
    try:
        # Save uploaded image temporarily
        temp_path = "/tmp/demo_image.jpg"
        image.save(temp_path)
        
        # Load and preprocess
        img_tensor = load_image(temp_path, mode=mode).to(DEVICE)
        
        # Generate
        with torch.no_grad():
            generated_ids = MODEL.generate(
                images=img_tensor,
                max_length=max_length,
                temperature=temperature,
                top_p=top_p
            )
        
        # Decode
        text = TOKENIZER.batch_decode(generated_ids, skip_special_tokens=True)[0]
        
        # Add statistics
        stats = f"\n\n---\nMode: {mode} ({MODE_CONFIGS[mode]['tokens']} tokens)\n"
        stats += f"Generated text length: {len(text)} characters\n"
        stats += f"Temperature: {temperature}, Top-p: {top_p}"
        
        return text + stats
    
    except Exception as e:
        return f"Error: {str(e)}"


def create_demo():
    """Create Gradio interface."""
    if not GRADIO_AVAILABLE:
        raise RuntimeError("Gradio not available")
    
    with gr.Blocks(title="DeepSeek-OCR Demo") as demo:
        gr.Markdown(
            """
            # DeepSeek-OCR: Contexts Optical Compression
            
            Upload an image to extract text using the DeepSeek-OCR model.
            
            **Paper:** [DeepSeek-OCR: Contexts Optical Compression](https://arxiv.org/abs/2510.18234)
            """
        )
        
        with gr.Row():
            with gr.Column():
                image_input = gr.Image(type="pil", label="Upload Image")
                
                mode_input = gr.Radio(
                    choices=list(MODE_CONFIGS.keys()),
                    value="base",
                    label="Resolution Mode",
                    info="Higher modes use more tokens but handle larger images better"
                )
                
                with gr.Accordion("Advanced Settings", open=False):
                    temperature_input = gr.Slider(
                        minimum=0.1,
                        maximum=2.0,
                        value=0.7,
                        step=0.1,
                        label="Temperature"
                    )
                    top_p_input = gr.Slider(
                        minimum=0.1,
                        maximum=1.0,
                        value=0.9,
                        step=0.05,
                        label="Top-p (Nucleus Sampling)"
                    )
                    max_length_input = gr.Slider(
                        minimum=256,
                        maximum=4096,
                        value=2048,
                        step=256,
                        label="Max Length"
                    )
                
                submit_btn = gr.Button("Run OCR", variant="primary")
            
            with gr.Column():
                output_text = gr.Textbox(
                    label="Extracted Text",
                    lines=20,
                    max_lines=30
                )
        
        gr.Markdown(
            """
            ### Mode Guide:
            - **Tiny** (64 tokens): Fast, for simple documents
            - **Small** (100 tokens): Balanced for most use cases
            - **Base** (256 tokens): Good quality for standard documents
            - **Large** (400 tokens): Best quality for complex layouts
            """
        )
        
        submit_btn.click(
            fn=run_ocr_demo,
            inputs=[
                image_input,
                mode_input,
                temperature_input,
                top_p_input,
                max_length_input
            ],
            outputs=output_text
        )
        
        # Examples
        gr.Examples(
            examples=[
                ["demo/examples/example1.jpg", "base", 0.7, 0.9, 2048],
                ["demo/examples/example2.png", "small", 0.7, 0.9, 2048],
            ],
            inputs=[
                image_input,
                mode_input,
                temperature_input,
                top_p_input,
                max_length_input
            ],
        )
    
    return demo


def parse_args():
    parser = argparse.ArgumentParser(description="DeepSeek-OCR Demo")
    parser.add_argument(
        "--checkpoint",
        type=str,
        default=None,
        help="Path to model checkpoint"
    )
    parser.add_argument(
        "--share",
        action="store_true",
        help="Create public link"
    )
    parser.add_argument(
        "--port",
        type=int,
        default=7860,
        help="Port to run on"
    )
    return parser.parse_args()


def main():
    args = parse_args()
    
    # Load model
    load_model(args.checkpoint)
    
    # Create and launch demo
    demo = create_demo()
    demo.launch(
        share=args.share,
        server_port=args.port,
        server_name="0.0.0.0"
    )


if __name__ == "__main__":
    main()
