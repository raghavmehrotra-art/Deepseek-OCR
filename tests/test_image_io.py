"""Tests for utils/image_io.py - covers Pillow (9.0.0 → 12.2.0) and torch usage."""
import io
import os
import tempfile
from unittest.mock import MagicMock, patch

import numpy as np
import pytest
import torch
from PIL import Image


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def _make_rgb_image(w: int = 200, h: int = 150) -> Image.Image:
    arr = np.random.randint(0, 255, (h, w, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def _save_temp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    fd, path = tempfile.mkstemp(suffix=suffix)
    os.close(fd)
    img.save(path)
    return path


# ---------------------------------------------------------------------------
# Import the module under test
# ---------------------------------------------------------------------------

import sys
from pathlib import Path
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_has_required_keys(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg, f"{mode} missing 'size'"
            assert "tokens" in cfg, f"{mode} missing 'tokens'"
            assert "pad" in cfg, f"{mode} missing 'pad'"

    def test_pad_flag_values(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_size_tuples(self):
        for mode, cfg in MODE_CONFIGS.items():
            size = cfg["size"]
            assert isinstance(size, tuple) and len(size) == 2


# ---------------------------------------------------------------------------
# resize_and_pad
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = _make_rgb_image(200, 150)
        result = resize_and_pad(img, (1024, 1024))
        assert result.size == (1024, 1024)

    def test_output_is_pil_rgb(self):
        img = _make_rgb_image(300, 200)
        result = resize_and_pad(img, (640, 640))
        assert isinstance(result, Image.Image)
        assert result.mode == "RGB"

    def test_square_input_no_padding_needed(self):
        img = _make_rgb_image(100, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_wide_image_letterboxed(self):
        img = _make_rgb_image(400, 100)  # very wide
        result = resize_and_pad(img, (256, 256))
        assert result.size == (256, 256)

    def test_custom_pad_color(self):
        img = _make_rgb_image(50, 100)
        result = resize_and_pad(img, (200, 200), pad_color=(0, 0, 0))
        assert result.size == (200, 200)

    def test_pillow_bilinear_resize_used(self):
        """Ensure Image.BILINEAR is accessible (Pillow API compatibility)."""
        assert hasattr(Image, "BILINEAR")


# ---------------------------------------------------------------------------
# load_image
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self):
        img = _make_rgb_image(300, 200)
        path = _save_temp_image(img)
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert isinstance(tensor, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_image):
        tensor = load_image(tmp_image, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_image):
        tensor = load_image(tmp_image, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_image):
        tensor = load_image(tmp_image, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert tensor.shape == (1, 3, h, w)

    def test_return_pil(self, tmp_image):
        result = load_image(tmp_image, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_tensor_dtype_float32(self, tmp_image):
        tensor = load_image(tmp_image, mode="tiny")
        assert tensor.dtype == torch.float32

    def test_unknown_mode_raises(self, tmp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_image, mode="ultra")

    def test_normalized_values_range(self, tmp_image):
        """After ImageNet normalization values are in a reasonable range."""
        tensor = load_image(tmp_image, mode="tiny")
        # Values should not be in [0, 1] anymore after normalization
        # but should be finite
        assert torch.isfinite(tensor).all()

    def test_png_support(self):
        img = _make_rgb_image(100, 100)
        path = _save_temp_image(img, suffix=".png")
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[0] == 1
        finally:
            os.unlink(path)

    def test_grayscale_converted_to_rgb(self):
        arr = np.random.randint(0, 255, (100, 100), dtype=np.uint8)
        img = Image.fromarray(arr, mode="L")
        path = _save_temp_image(img)
        try:
            tensor = load_image(path, mode="tiny")
            assert tensor.shape[1] == 3  # RGB channels
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_same_size_returns_total(self):
        result = calculate_valid_tokens(1024, 1024, 1024, 1024, 256)
        assert result == 256

    def test_smaller_image_returns_fewer_tokens(self):
        result = calculate_valid_tokens(512, 512, 1024, 1024, 256)
        assert result < 256
        assert result > 0

    def test_wide_image(self):
        result = calculate_valid_tokens(2048, 512, 1024, 1024, 256)
        assert isinstance(result, int)
        assert 0 < result <= 256

    def test_tall_image(self):
        result = calculate_valid_tokens(512, 2048, 1024, 1024, 256)
        assert isinstance(result, int)
        assert 0 < result <= 256

    def test_returns_int(self):
        result = calculate_valid_tokens(800, 600, 1024, 1024, 400)
        assert isinstance(result, int)

    def test_zero_tokens_edge(self):
        result = calculate_valid_tokens(1, 1, 1024, 1024, 256)
        # Very small image → very few valid tokens (could be 0)
        assert result >= 0
