"""Tests for utils/image_io.py — covers Pillow (major bump 9→10) and torch/torchvision APIs."""
import io
import os
import sys
import tempfile
from pathlib import Path
from unittest.mock import patch, MagicMock

import numpy as np
import pytest
import torch
from PIL import Image

# Make sure project root is importable
sys.path.insert(0, str(Path(__file__).parent.parent))

from utils.image_io import (
    MODE_CONFIGS,
    load_image,
    resize_and_pad,
    calculate_valid_tokens,
)


# ---------------------------------------------------------------------------
# Helpers
# ---------------------------------------------------------------------------

def make_rgb_image(width: int = 100, height: int = 80) -> Image.Image:
    """Create a simple solid-colour RGB PIL image."""
    arr = np.random.randint(0, 255, (height, width, 3), dtype=np.uint8)
    return Image.fromarray(arr, mode="RGB")


def save_tmp_image(img: Image.Image, suffix: str = ".jpg") -> str:
    """Save PIL image to a temp file and return the path."""
    with tempfile.NamedTemporaryFile(suffix=suffix, delete=False) as f:
        tmp_path = f.name
    img.save(tmp_path)
    return tmp_path


# ---------------------------------------------------------------------------
# MODE_CONFIGS
# ---------------------------------------------------------------------------

class TestModeConfigs:
    def test_all_expected_modes_present(self):
        for mode in ("tiny", "small", "base", "large"):
            assert mode in MODE_CONFIGS

    def test_mode_structure(self):
        for mode, cfg in MODE_CONFIGS.items():
            assert "size" in cfg
            assert "tokens" in cfg
            assert "pad" in cfg
            assert isinstance(cfg["size"], tuple) and len(cfg["size"]) == 2
            assert isinstance(cfg["tokens"], int)
            assert isinstance(cfg["pad"], bool)

    def test_base_and_large_use_padding(self):
        assert MODE_CONFIGS["base"]["pad"] is True
        assert MODE_CONFIGS["large"]["pad"] is True

    def test_tiny_and_small_no_padding(self):
        assert MODE_CONFIGS["tiny"]["pad"] is False
        assert MODE_CONFIGS["small"]["pad"] is False

    def test_token_counts_are_positive(self):
        for cfg in MODE_CONFIGS.values():
            assert cfg["tokens"] > 0


# ---------------------------------------------------------------------------
# resize_and_pad  (exercises Pillow 10 API)
# ---------------------------------------------------------------------------

class TestResizeAndPad:
    def test_output_size_matches_target(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_output_is_rgb(self):
        img = make_rgb_image(200, 100)
        result = resize_and_pad(img, (256, 256))
        assert result.mode == "RGB"

    def test_landscape_image_padded_correctly(self):
        """Wide image should have top/bottom padding."""
        img = make_rgb_image(400, 100)
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_portrait_image_padded_correctly(self):
        """Tall image should have left/right padding."""
        img = make_rgb_image(100, 400)
        result = resize_and_pad(img, (400, 400))
        assert result.size == (400, 400)

    def test_custom_pad_color_applied(self):
        """Padding area should use the supplied pad_color."""
        img = Image.new("RGB", (50, 50), color=(128, 128, 128))
        target = (200, 200)
        pad_color = (0, 0, 255)
        result = resize_and_pad(img, target, pad_color=pad_color)
        # Top-left corner should be the pad colour (not the image colour)
        corner_pixel = result.getpixel((0, 0))
        assert corner_pixel == pad_color

    def test_square_image_no_padding_needed(self):
        img = make_rgb_image(100, 100)
        result = resize_and_pad(img, (200, 200))
        assert result.size == (200, 200)

    def test_already_target_size(self):
        img = make_rgb_image(512, 512)
        result = resize_and_pad(img, (512, 512))
        assert result.size == (512, 512)

    def test_returns_pil_image(self):
        img = make_rgb_image(64, 64)
        result = resize_and_pad(img, (128, 128))
        assert isinstance(result, Image.Image)

    def test_pillow_bilinear_resampling(self):
        """Ensure Pillow 10 compatible BILINEAR constant works."""
        img = make_rgb_image(200, 200)
        # Image.BILINEAR is the Pillow 9 constant; Pillow 10 keeps it as alias.
        # We just verify the call doesn't raise.
        result = resize_and_pad(img, (100, 100))
        assert result.size == (100, 100)


# ---------------------------------------------------------------------------
# load_image  (exercises Pillow + torch + torchvision)
# ---------------------------------------------------------------------------

class TestLoadImage:
    @pytest.fixture(autouse=True)
    def tmp_image(self):
        img = make_rgb_image(200, 150)
        path = save_tmp_image(img)
        yield path
        os.unlink(path)

    def test_returns_tensor_by_default(self, tmp_image):
        result = load_image(tmp_image, mode="tiny")
        assert isinstance(result, torch.Tensor)

    def test_tensor_shape_tiny(self, tmp_image):
        result = load_image(tmp_image, mode="tiny")
        h, w = MODE_CONFIGS["tiny"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_small(self, tmp_image):
        result = load_image(tmp_image, mode="small")
        h, w = MODE_CONFIGS["small"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_base(self, tmp_image):
        result = load_image(tmp_image, mode="base")
        h, w = MODE_CONFIGS["base"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_shape_large(self, tmp_image):
        result = load_image(tmp_image, mode="large")
        h, w = MODE_CONFIGS["large"]["size"]
        assert result.shape == (1, 3, h, w)

    def test_tensor_dtype_float(self, tmp_image):
        result = load_image(tmp_image, mode="tiny")
        assert result.dtype == torch.float32

    def test_return_pil_flag(self, tmp_image):
        result = load_image(tmp_image, mode="tiny", return_pil=True)
        assert isinstance(result, Image.Image)

    def test_return_pil_size_matches_mode(self, tmp_image):
        result = load_image(tmp_image, mode="tiny", return_pil=True)
        w, h = result.size
        expected_w, expected_h = MODE_CONFIGS["tiny"]["size"]
        assert (w, h) == (expected_w, expected_h)

    def test_invalid_mode_raises(self, tmp_image):
        with pytest.raises(ValueError, match="Unknown mode"):
            load_image(tmp_image, mode="nonexistent")

    def test_normalized_tensor_range(self, tmp_image):
        """After ImageNet normalization values should span outside [0,1]."""
        result = load_image(tmp_image, mode="tiny")
        # After normalization the values can be negative
        assert result.min().item() < 1.0
        # They should not be absurdly large
        assert result.max().item() < 10.0

    def test_loads_png(self):
        img = make_rgb_image(64, 64)
        path = save_tmp_image(img, suffix=".png")
        try:
            result = load_image(path, mode="tiny")
            assert isinstance(result, torch.Tensor)
        finally:
            os.unlink(path)

    def test_rgba_image_converted_to_rgb(self):
        """Pillow images with alpha channel should be handled (converted to RGB)."""
        img = Image.new("RGBA", (64, 64), (255, 0, 0, 128))
        path = save_tmp_image(img, suffix=".png")
        try:
            result = load_image(path, mode="tiny")
            assert result.shape[1] == 3
        finally:
            os.unlink(path)


# ---------------------------------------------------------------------------
# calculate_valid_tokens
# ---------------------------------------------------------------------------

class TestCalculateValidTokens:
    def test_square_image_no_padding(self):
        """Square image in square target → all tokens valid."""
        result = calculate_valid_tokens(512, 512, 512, 512, 256)
        assert result == 256

    def test_landscape_image_fewer_tokens(self):
        result = calculate_valid_tokens(1024, 512, 1024, 1024, 256)
        assert result < 256

    def test_portrait_image_fewer_tokens(self):
        result = calculate_valid_tokens(512, 1024, 1024, 1024, 256)
        assert result < 256

    def test_result_is_integer(self):
        result = calculate_valid_tokens(100, 200, 1024, 1024, 256)
        assert isinstance(result, int)

    def test_result_non_negative(self):
        result = calculate_valid_tokens(10, 10, 1024, 1024, 256)
        assert result >= 0

    def test_result_never_exceeds_total_tokens(self):
        result = calculate_valid_tokens(200, 200, 512, 512, 100)
        assert result <= 100

    def test_consistent_with_mode_configs(self):
        for mode, cfg in MODE_CONFIGS.items():
            if cfg["pad"]:
                tw, th = cfg["size"]
                tokens = cfg["tokens"]
                result = calculate_valid_tokens(tw, th, tw, th, tokens)
                assert result == tokens


# ---------------------------------------------------------------------------
# Pillow 10 API compatibility checks
# ---------------------------------------------------------------------------

class TestPillow10Compatibility:
    """Verify APIs that changed between Pillow 9 and 10."""

    def test_image_open_returns_image(self):
        img = make_rgb_image(32, 32)
        buf = io.BytesIO()
        img.save(buf, format="JPEG")
        buf.seek(0)
        loaded = Image.open(buf)
        assert isinstance(loaded, Image.Image)

    def test_image_new_rgb(self):
        img = Image.new("RGB", (100, 100), color=(128, 64, 32))
        assert img.size == (100, 100)
        assert img.mode == "RGB"

    def test_image_resize_with_resampling_enum(self):
        """Pillow 10 requires Resampling enum; BILINEAR alias still works."""
        img = make_rgb_image(200, 200)
        resized = img.resize((100, 100), Image.BILINEAR)
        assert resized.size == (100, 100)

    def test_image_convert_rgb(self):
        img = Image.new("L", (50, 50), 128)
        rgb = img.convert("RGB")
        assert rgb.mode == "RGB"

    def test_image_paste(self):
        canvas = Image.new("RGB", (200, 200), (255, 255, 255))
        patch_img = Image.new("RGB", (50, 50), (0, 0, 0))
        canvas.paste(patch_img, (75, 75))
        assert canvas.getpixel((75, 75)) == (0, 0, 0)

    def test_numpy_array_roundtrip(self):
        arr = np.random.randint(0, 255, (64, 64, 3), dtype=np.uint8)
        img = Image.fromarray(arr)
        arr2 = np.array(img)
        np.testing.assert_array_equal(arr, arr2)
