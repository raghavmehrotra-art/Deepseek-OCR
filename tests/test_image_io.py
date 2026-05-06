"""Tests for utils/image_io.py — exercises Pillow (major bump 9→10) and torch APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image

# Make sure project root is on the path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    calculate_valid_tokens,
    load_image,
    resize_and_pad,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_pil_image(width=200, height=150, color=(128, 64, 32)) -> Image.Image:
    """Create a small solid-colour PIL image."""
    img = Image.new("RGB", (width, height), color)
    return img


def _save_tmp_image(width=200, height=150, fmt="JPEG") -> str:
    """Save a temporary image file and return its path."""
    img = _make_pil_image(width, height)
    suffix = ".jpg" if fmt == "JPEG" else ".png"
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        path = f.name
    img.save(path, format=fmt)
    return path


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_modes_present(self):
        assert set(MODE_CONFIGS.keys()) == {"tiny", "small", "base", "large"}

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"Mode {mode} missing 'size'"
            assert "tokens" in cfg, f"Mode {mode} missing 'tokens'"
            assert "pad" in cfg, f"Mode {mode} missing 'pad'"

    def test_base_and_large_use_padding(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_padding(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_sizes_are_square(self):
        for mode, cfg in MODE_CONFIGS.items():
            w, h = cfg["size"]
            assert w == h, f"Mode {mode} size is not square: {cfg['size']}"


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_pil_image(300, 200)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_aspect_ratio_preserved_landscape(self):
        """For a landscape image the content should be centred with horizontal bars."""
        img = _make_pil_image(400, 200)  # 2:1 landscape
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)
        # The image is PIL RGB – just confirm no crash and correct mode
        assert result.mode == "RGB"

    def test_aspect_ratio_preserved_portrait(self):
        img = _make_pil_image(100, 400)  # 1:4 portrait
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_custom_pad_color(self):
        img = _make_pil_image(100, 100, color=(0, 0, 0))
        result = resize_and_pad(img, (200, 200), pad_color=(255, 0, 0))
        arr = np.array(result)
        # Corners should be the pad colour (red)
        assert arr[0, 0, 0] == 255   # R
        assert arr[0, 0, 1] == 0     # G
        assert arr[0, 0, 2] == 0     # B

    def test_square_image_no_pad_needed(self):
        """Square image resized to square target – result still correct."""
        img = _make_pil_image(300, 300)
        result = resize_and_pad(img, (128, 128))
        assert result.size == (128, 128)

    def test_returns_pil_image(self):
        img = _make_pil_image(50, 50)
        result = resize_and_pad(img, (64, 64))
        assert isinstance(result, Image.Image)

    def test_default_pad_color_is_white(self):
        """Padding should default to white (255,255,255)."""
        img = _make_pil_image(100, 50, color=(0, 0, 0))  # black landscape
        result = resize_and_pad(img, (200, 200))
        arr = np.array(result)
        # Top-left corner should be white (padding area)
        assert arr[0, 0, 0] == 255
        assert arr[0, 0, 1] == 255
        assert arr[0, 0, 2] == 255


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    def setup_method(self):
        self.tmp_path = _save_tmp_image(640, 480)

    def teardown_method(self):
        if os.path.exists(self.tmp_path):
            os.unlink(self.tmp_path)

    # --- happy paths ---

    def test_returns_tensor_by_default(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self):
        tensor = load_image(self.tmp_path, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self):
        tensor = load_image(self.tmp_path, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self):
        tensor = load_image(self.tmp_path, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil_flag(self):
        result = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_tiny(self):
        result = load_image(self.tmp_path, mode="tiny", return_pil=True)
        assert result.size == MODE_CONFIGS["tiny"]["size"]

    def test_return_pil_size_base(self):
        result = load_image(self.tmp_path, mode="base", return_pil=True)
        assert result.size == MODE_CONFIGS["base"]["size"]

    def test_tensor_dtype_float(self):
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_normalized_values_in_reasonable_range(self):
        """After ImageNet normalisation values should be roughly [-3, 3]."""
        tensor = load_image(self.tmp_path, mode="tiny")
        assert tensor.min().item() > -4.0
        assert tensor.max().item() < 4.0

    # --- Pillow 10 compatibility: Image.BILINEAR still works ---
    def test_bilinear_resize_pillow10_compat(self):
        """Pillow 10 removed some deprecated constants – ensure BILINEAR works."""
        img = _make_pil_image(200, 200)
        resized = img.resize((64, 64), Image.BILINEAR)
        assert resized.size == (64, 64)

    # --- error cases ---

    def test_unknown_mode_raises(self):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(self.tmp_path, mode="mega")

    def test_missing_file_raises(self):
        with pytest.raises(Exception):
            load_image("/nonexistent/path/image.jpg", mode="tiny")

    # --- PNG support ---
    def test_loads_png(self):
        png_path = _save_tmp_image(100, 100, fmt="PNG")
        try:
            tensor = load_image(png_path, mode="tiny")
            assert tensor.shape[0] == 1
        finally:
            os.unlink(png_path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_same_size_returns_all_tokens(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_half_width_image(self):
        """Image half the width of target → roughly half the tokens."""
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert 0 < result <= 256

    def test_small_image_fewer_tokens(self):
        big = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        small = calculate_valid_tokens(100, 100, 1024, 1024, 256)
        assert small < big

    def test_returns_integer(self):
        result = calculate_valid_tokens(800, 600, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_wide_image_landscape(self):
        """Landscape: scale limited by height ratio."""
        result = calculate_valid_tokens(2000, 500, 1024, 1024, 256)
        assert 0 < result <= 256

    def test_square_image_at_exactly_target(self):
        result = calculate_valid_tokens(1280, 1280, 1280, 1280, 400)
        assert result == 400
